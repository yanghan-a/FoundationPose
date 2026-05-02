"""FoundationPose model-free pipeline.

End-to-end: capture multi-view RGBD of an object with the same camera you
will use at inference, estimate per-frame object pose (Charuco board by
default), then call BundleSDF's NeuralObjectField to reconstruct a
textured mesh that drops straight into the existing model-based scripts
(`track_single.py` / `track_anything.py`).
"""
