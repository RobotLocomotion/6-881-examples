(define (stream manipulation-station)

  (:stream sample-reachable-grasp
    :inputs (?r ?o ?p)
    :domain (and (Robot ?r) (InitPose ?o ?p))
    :outputs (?g ?q ?t)
    :certified (and (Grasp ?o ?g) (Conf ?r ?q) (Traj ?t)
                    (Kin ?r ?o ?p ?g ?q ?t))
  )

  (:stream sample-reachable-pose
    :inputs (?r ?o ?g ?s)
    :domain (and (Robot ?r) (Grasp ?o ?g) (Stackable ?o ?s))
    :outputs (?p ?q ?t)
    :certified (and (Pose ?o ?p) (Supported ?o ?p ?s) (Conf ?r ?q) (Traj ?t)
                    (Kin ?r ?o ?p ?g ?q ?t))
  )

  (:stream plan-pull
    :inputs (?robot ?door ?door_conf1 ?door_conf2)
    :domain (and (Robot ?robot) (Conf ?door ?door_conf1) (Conf ?door ?door_conf2) (Door ?door))
    :outputs (?robot_conf1 ?robot_conf2 ?traj)
    :certified (and (Conf ?robot ?robot_conf1) (Conf ?robot ?robot_conf2) (Traj ?traj)
                    (Pull ?robot ?robot_conf1 ?robot_conf2 ?door ?door_conf1 ?door_conf2 ?traj))
  )

  (:stream plan-motion
    :inputs (?robot ?conf1 ?conf2)
    :domain (and (Conf ?robot ?conf1) (Conf ?robot ?conf2) (Robot ?robot))
    ; Advanced feature that considers a subset of the fluent state as an input
    :fluents (AtConf AtPose AtGrasp)
    :outputs (?traj)
    :certified (Motion ?robot ?conf1 ?conf2 ?traj)
  )

  ; Boolean functions (i.e. predicates) that are similar to test streams
  (:predicate (TrajPoseCollision ?traj ?obj ?pose)
    (and (Traj ?traj) (Pose ?obj ?pose))
  )

  (:predicate (TrajConfCollision ?traj ?d ?conf)
    (and (Traj ?traj) (Conf ?d ?conf))
  )
)
