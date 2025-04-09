import open3d as o3d
import numpy as np


def visualize_comparison(original, simplified):
    pcd_orig = o3d.geometry.PointCloud()
    pcd_orig.points = o3d.utility.Vector3dVector(original)
    pcd_orig.paint_uniform_color([0, 0, 1])

    pcd_simp = o3d.geometry.PointCloud()
    pcd_simp.points = o3d.utility.Vector3dVector(simplified)
    pcd_simp.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd_orig, pcd_simp])