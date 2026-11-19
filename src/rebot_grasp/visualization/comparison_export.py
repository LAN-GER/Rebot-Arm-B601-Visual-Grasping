"""Write the image, metric depth diagnostics and paired point clouds of one snapshot."""
import json
from pathlib import Path

import numpy as np

from rebot_grasp.visualization.cloud_windows import diagnostic_image


def save_comparison_files(output, capture, backend, result, comparison, image):
    import cv2

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
    images = {'comparison.png': image,
              'color.png': capture.frame.color_bgr,
              'diagnostics.png': diagnostic_image(capture, backend, result)}
    prepared = backend.last_prepared
    if prepared is not None:
        for name, cloud in (('pointcloud_raw.npz', prepared.raw), ('pointcloud.npz', prepared.cloud)):
            np.savez_compressed(output / name, points_m=cloud.points, colors_rgb=cloud.colors_rgb,
                                target_mask=cloud.target_mask)
        images.update({'target_mask.png': backend.last_target.mask.astype(np.uint8) * 255,
                       'eligible_target.png': backend.last_eligible_mask.astype(np.uint8) * 255,
                       'gripper_exclusion.png': prepared.gripper_mask.astype(np.uint8) * 255})
    for name, image in images.items():
        if not cv2.imwrite(str(output / name), image):
            raise OSError(f'Could not save {name}')
