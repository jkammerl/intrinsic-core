# Visualize the Solution

In this tutorial, you'll observe the execution of the Open Machine Tending
Solution (OMTS) using a visualizer.

## Prerequisites

To detect the position of the workpiece, this Solution runs ML models. With an
NVIDIA GPU (set up with `setup_nvidia.sh`, see [Getting started](getting_started.md))
this takes a few seconds. Without one, the models run on the CPU, which works
but takes a few minutes per detection.

If you haven't already, follow [Getting started](getting_started.md) to start OMTS, then [Visualize the robot](visualize_the_robot.md) to start RViz and/or Gazebo.

## Start the Process

The various steps of the OMTS are defined in
[main.py](https://github.com/intrinsic-ai/intrinsic-omts/blob/main/src/main.py)
and its dependencies.

1. Configure the pose estimator used to detect the position of the raw stock
   workpiece from a camera image:

   ```shell
   cd ~/intrinsic-omts
   bazel run //tools/pose_estimation:register_using_train_service -- \
      --address="localhost:17080" \
      --scene_object_id="ai.intrinsic.raw_stock_2x3x5" \
      --pose_estimator_id="ai.intrinsic.raw_stock_2x3x5_estimator" \
      --refinement_iters=6 \
      --confidence_threshold=0.6 \
      --visibility_threshold=0.6
   ```

   You should see:

   ```
   Complete! Pose estimator ai.intrinsic.raw_stock_2x3x5_estimator saved.
   ```

   _This command should succeed quickly. Despite the references to "training
   jobs", this doesn't do any training. It just records the geometry of the
   workpiece so that the pose estimator "knows what to look for", so to speak.
   All the real work is done after the camera image is captured._

1. Run the Solution itself. The `--config` flag selects the cell configuration
   matching the `--config=lab_bb_01` flag you used when deploying:

   ```shell
   cd ~/intrinsic-omts
   bazel run //src:omts_app -- \
      --address=localhost:17080 \
      --config=configs/lab_bb_01/app_config.yaml
   ```

If you get an error, check the [Troubleshooting](#troubleshooting) section below.
Otherwise, while it's running, try to spot the following steps:

- **Pose estimation**: The robot points the wrist camera at the workpiece,
  captures an image, then uses the pose estimator to determine the coordinates
  of the workpiece in the image. It then relies on its calibration to map these
  to a position within the belief world that it can use to grasp the workpiece.
- **Sensor-based control**: The robot slowly descends towards workpiece. Using
  real-time force sensor feedback, it stops when it reaches the part.
- **Motion planning**: Intrinsic Core uses knowledge of the geometry of the
  robot, workpiece, enclosure and CNC machine to move the workpiece from the
  table to the vise without collisions.

### Troubleshooting

| Symptom | Cause | Resolution |
| :--- | :--- | :--- |
| `StatusCode: ai.intrinsic.dio_set_output:14 User Report: no healthy upstream` | The Solution was deployed with the default `--operation_mode=real`, but this PC has not been set up for real-time execution, so ICON doesn't run and the `dio_set_output` skill can't reach it. | To run in simulation, redeploy:<br>`inctl solution stop --address localhost:17080`<br>`bazel run //:omts_solution --config=lab_bb_01 -- --address localhost:17080 --operation_mode=sim`<br>To run on a real robot, run `sudo ~/intrinsic-core/intrinsic_runtime/setup_realtime.sh`, then reboot. |
| `StatusCode: ai.intrinsic.estimate_pose_multi_view:13 User Report: [...] status = StatusCode.NOT_FOUND details = "requested id "ai.intrinsic.raw_stock_2x3x5_estimator" not found" [...]` | The pose estimator has not been loaded into the Solution since it was last started or restarted. | Rerun the `register_using_train_service` command from [Start the Process](#start-the-process) above. |
| `StatusCode: ai.intrinsic.estimate_pose_multi_view:13 User Report: Pose estimation resulted in 0 detections. Expected at least 1 instances.` | The pose estimator did not detect the raw stock workpiece in the rendered or captured image. This can happen if noise in the physics simulator moved the workpiece out of view of the simulated camera. | Stop and restart the Solution:<br>`inctl solution stop --address localhost:17080`<br>`bazel run //:omts_solution --config=lab_bb_01 -- --address localhost:17080 --operation_mode=sim` |
| The robot pauses for longer than a minute with the camera facing the raw stock workpiece. After 30 minutes, `omts_app` fails with `status = StatusCode.NOT_FOUND`. | A race condition on startup means that the pose estimator was not properly loaded. | Restart the inference service, then retry:<br>`kubectl -n app-resources delete pod rs-inference-service-0` |
| The robot pauses for a few minutes with the camera facing the raw stock workpiece. | The ML models run on the CPU, because this PC either lacks an NVIDIA GPU, or the GPU has not been integrated with the k3s cluster. The inference service logs `Created ONNX Runtime session with ['CPUExecutionProvider']`. | If the PC has an NVIDIA GPU, run `sudo ~/intrinsic-core/intrinsic_runtime/setup_nvidia.sh`, then restart the inference service: `kubectl -n app-resources delete pod rs-inference-service-0`. Otherwise, wait: detection completes on the CPU. |
| `StatusCode: ai.intrinsic.move_robot:10201 User Report: Error reported when planning for motion segment 0. Invalid initial joint configuration. Collision reported: Config: [-55.0365,-87.6166,-135.024,-47.3632,89.9952,304.13] in deg, Left entity: gripper.gripper_finger2 Right entities: enclosure.base_link` | This can happen in simulation because the physics simulator doesn't model the grasping force of the vise, so the workpiece can fall out of the vise and the robot fails to pick it up after the CNC machine is done. Normally the Process would check for a successful grasp, but without that check it moves back to the table with an empty gripper, finally bumping the gripper (`gripper_finger2` here) into the enclosure's `base_link` (the table). | Reset to the starting state before retrying, which brings the workpiece back to a known position: `inctl world reset --address localhost:17080`. In a real workcell, you'd also need to remove the misplaced workpiece from the CNC machine. |
| `StatusCode: ai.intrinsic.move_robot:10601 User Report: Frame with name "pre_place_vise" does not exist on object with id="root" and name="root".; Unable to resolve TransformNodeReference [id: ];` | The frame names in the workcell selected when deploying `//:omts_solution` don't match the frame names in the workcell selected when running `//src:omts_app`. | Either redeploy the Solution or rerun the Process, making sure that `--config=...` matches in both. |

## What's next

Continue to the [Cell customization](cell_customization.md) tutorial and try
using a similar approach to adjust the frame used when returning the part after
processing: Can you reduce the time needed to put it down?
