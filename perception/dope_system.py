import numpy as np
import cv2
import os
import yaml

from PIL import Image, ImageDraw

from pydrake.common.eigen_geometry import Quaternion
from pydrake.examples.manipulation_station import (ManipulationStation,
                                                   CreateDefaultYcbObjectList)
from pydrake.math import RigidTransform
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (AbstractValue, BasicVector,
                                       DiagramBuilder, LeafSystem)
from pydrake.systems.rendering import PoseBundle
from pydrake.systems.sensors import ImageRgba8U

from inference.cuboid import Cuboid3d
from inference.cuboid_pnp_solver import CuboidPNPSolver
from inference.detector import ModelData, ObjectDetector


class DopeSystem(LeafSystem):
    """
    A system that runs DOPE (https://github.com/NVlabs/Deep_Object_Pose). Note
    that this only runs the inference step of DOPE, so trained weights must be
    provided. Given a single RGB image, DOPE will compute poses of the
    supported YCB objects and produce an image with bounding boxes drawn over
    all detected objects. See the DOPE documentation for more information.

    @system{
      @input_port{rgb_input_image},
      @output_port{annotated_rgb_image},
      @output_port{pose_bundle}
    """
    def __init__(self, weights_path, config_file, X_WCamera):
        """
        @param weights a path to a directory containing model weights.
        @param config_file a path to a .yaml file with the DOPE configuration.
        """
        LeafSystem.__init__(self)

        self.params = None
        with open(config_file, 'r') as stream:
            try:
                print(
                    "Loading DOPE parameters from '{}'...".format(config_file))
                self.params = yaml.load(stream)
                print('    Parameters loaded.')
            except yaml.YAMLError as exc:
                print(exc)

        self._SetupSolvers(weights_path)

        self.model_names = self.models.keys()
        self.poses = {}
        self.image_data = None
        self.g_draw = None

        # TODO(kmuhlrad): update documentation
        self.X_WCamera = X_WCamera

        # TODO(kmuhlrad): Use camera config file instead of hardcoded values.
        camera_width = 848
        camera_height = 480

        self.rgb_input_image = self.DeclareAbstractInputPort(
            "rgb_input_image", AbstractValue.Make(
                ImageRgba8U(camera_width, camera_height)))

        self.DeclareAbstractOutputPort("annotated_rgb_image",
                                        lambda: AbstractValue.Make(
                                            ImageRgba8U(camera_width,
                                                        camera_height)),
                                        self._DoCalcAnnotatedImage)

        self.DeclareAbstractOutputPort("pose_bundle",
                                        lambda: AbstractValue.Make(
                                            PoseBundle(num_poses=len(
                                                self.model_names))),
                                        self._DoCalcPoseBundle)

    def _SetupSolvers(self, weights_path):
        self.models = {}
        self.pnp_solvers = {}
        self.draw_colors = {}

        # Initialize parameters
        matrix_camera = np.zeros((3,3))
        matrix_camera[0, 0] = self.params["camera_settings"]['fx']
        matrix_camera[1, 1] = self.params["camera_settings"]['fy']
        matrix_camera[0, 2] = self.params["camera_settings"]['cx']
        matrix_camera[1, 2] = self.params["camera_settings"]['cy']
        matrix_camera[2, 2] = 1
        dist_coeffs = np.zeros((4,1))

        if "dist_coeffs" in self.params["camera_settings"]:
            dist_coeffs = np.array(
                self.params["camera_settings"]['dist_coeffs'])

        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = self.params['thresh_angle']
        self.config_detect.thresh_map = self.params['thresh_map']
        self.config_detect.sigma = self.params['sigma']
        self.config_detect.thresh_points = self.params["thresh_points"]

        # For each object to detect, load network model, and create PNP solver.
        for model in self.params['weights']:
            self.models[model] = \
                ModelData(
                    model,
                    os.path.join(weights_path, self.params['weights'][model])
                )
            self.models[model].load_net_model()

            self.draw_colors[model] = \
                tuple(self.params["draw_colors"][model])
            self.pnp_solvers[model] = \
                CuboidPNPSolver(
                    model,
                    matrix_camera,
                    Cuboid3d(self.params['dimensions'][model]),
                    dist_coeffs=dist_coeffs
                )

    def _DrawLine(self, point1, point2, lineColor, lineWidth):
        '''Draws line on image.'''
        if not point1 is None and point2 is not None:
            self.g_draw.line([point1,point2], fill=lineColor, width=lineWidth)

    def _DrawDot(self, point, pointColor, pointRadius):
        '''Draws dot (filled circle) on image.'''
        if point is not None:
            xy = [
                point[0] - pointRadius,
                point[1] - pointRadius,
                point[0] + pointRadius,
                point[1] + pointRadius
            ]
            self.g_draw.ellipse(xy,
                           fill=pointColor,
                           outline=pointColor
                           )

    def _DrawCube(self, points, color=(255, 0, 0)):
        '''
        Draws a cube with a thick solid line across the front top edge and an X
        on the top face.
        '''

        lineWidthForDrawing = 2

        # Draw the front.
        self._DrawLine(points[0], points[1], color, lineWidthForDrawing)
        self._DrawLine(points[1], points[2], color, lineWidthForDrawing)
        self._DrawLine(points[3], points[2], color, lineWidthForDrawing)
        self._DrawLine(points[3], points[0], color, lineWidthForDrawing)

        # Draw the back.
        self._DrawLine(points[4], points[5], color, lineWidthForDrawing)
        self._DrawLine(points[6], points[5], color, lineWidthForDrawing)
        self._DrawLine(points[6], points[7], color, lineWidthForDrawing)
        self._DrawLine(points[4], points[7], color, lineWidthForDrawing)

        # Draw the sides.
        self._DrawLine(points[0], points[4], color, lineWidthForDrawing)
        self._DrawLine(points[7], points[3], color, lineWidthForDrawing)
        self._DrawLine(points[5], points[1], color, lineWidthForDrawing)
        self._DrawLine(points[2], points[6], color, lineWidthForDrawing)

        # Draw the dots.
        self._DrawDot(points[0], pointColor=color, pointRadius=4)
        self._DrawDot(points[1], pointColor=color, pointRadius=4)

        # Draw an "X" on the top.
        self._DrawLine(points[0], points[5], color, lineWidthForDrawing)
        self._DrawLine(points[1], points[4], color, lineWidthForDrawing)


    def _RunDope(self, context):
        g_img = self.EvalAbstractInput(
            context,
            self.rgb_input_image.get_index()).get_value().data[:, :, :3]

        # Copy and draw image.
        img_copy = g_img.copy()
        im = Image.fromarray(img_copy)
        self.g_draw = ImageDraw.Draw(im)

        for m in self.models:
            # Detect object.
            results = ObjectDetector.detect_object_in_image(
                self.models[m].net,
                self.pnp_solvers[m],
                g_img,
                self.config_detect
            )

            # Publish pose and overlay cube on image.
            for i_r, result in enumerate(results):
                if result["location"] is None:
                    continue
                loc = result["location"]
                ori = result["quaternion"]

                CONVERT_SCALE_CM_TO_METERS = 100
                X_WObject = RigidTransform(
                    Quaternion(ori[3], ori[0], ori[1], ori[2]),
                    [loc[0] / CONVERT_SCALE_CM_TO_METERS,
                     loc[1] / CONVERT_SCALE_CM_TO_METERS,
                     loc[2] / CONVERT_SCALE_CM_TO_METERS])

                self.poses[m] = X_WObject.GetAsIsometry3()

                # Draw the cube.
                if None not in result['projected_points']:
                    points2d = []
                    for pair in result['projected_points']:
                        points2d.append(tuple(pair))
                    self._DrawCube(points2d, self.draw_colors[m])

        self.image_data = np.array(im)

    def _DoCalcPoseBundle(self, context, output):
        self._RunDope(context)

        for model in self.poses:
            output.get_mutable_value().set_pose(
                self.model_names.index(model), self.X_WCamera.multiply(self.poses[model]))
            output.get_mutable_value().set_name(
                self.model_names.index(model), model)

    def _DoCalcAnnotatedImage(self, context, output):
        self._RunDope(context)

        for i in range(self.image_data.shape[0]):
            for j in range(self.image_data.shape[1]):
                output.get_mutable_value().at(j, i)[:3] = (
                    self.image_data[i, j, :3])


if __name__ == "__main__":
    builder = DiagramBuilder()

    # Create the DopeSystem.
    weights_path = '/home/amazon/catkin_ws/src/dope/weights'
    config_file = '/home/amazon/catkin_ws/src/dope/config/config_pose.yaml'
    dope_system = builder.AddSystem(DopeSystem(weights_path, config_file))

    # Create the ManipulationStation.
    station = builder.AddSystem(ManipulationStation())
    station.SetupClutterClearingStation()
    ycb_objects = CreateDefaultYcbObjectList()
    for model_file, X_WObject in ycb_objects:
        station.AddManipulandFromFile(model_file, X_WObject)
    station.Finalize()

    # Connect the rgb images to the DopeSystem.
    builder.Connect(station.GetOutputPort("camera_0_rgb_image"),
                    dope_system.GetInputPort("rgb_input_image"))

    diagram = builder.Build()
    simulator = Simulator(diagram)

    context = diagram.GetMutableSubsystemContext(
        dope_system, simulator.get_mutable_context())

    # Check the poses.
    pose_bundle = dope_system.GetOutputPort("pose_bundle").Eval(context)
    for i in range(pose_bundle.get_num_poses()):
        if pose_bundle.get_name(i):
            print pose_bundle.get_name(i), pose_bundle.get_pose(i)

    # Show the annotated image.
    annotated_image = dope_system.GetOutputPort(
        "annotated_rgb_image").Eval(context).data
    cv2.imshow("dope image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)