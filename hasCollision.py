import sys
import numpy as np
import trimesh
from collections import deque
import time
from numba import jit, njit
import multiprocessing as mp
from functools import partial

# 使用numba加速数值计算


@njit
def bbox_intersects_numba(min_max1, min_max2):
    """numba加速的包围盒相交检测"""
    # 检查第一个包围盒的最小值是否小于等于第二个包围盒的最大值
    first = (min_max1[0, 0] <= min_max2[1, 0] and
             min_max1[0, 1] <= min_max2[1, 1] and
             min_max1[0, 2] <= min_max2[1, 2])

    # 检查第二个包围盒的最小值是否小于等于第一个包围盒的最大值
    second = (min_max2[0, 0] <= min_max1[1, 0] and
              min_max2[0, 1] <= min_max1[1, 1] and
              min_max2[0, 2] <= min_max1[1, 2])

    return first and second


@njit
def triangle_bbox_numba(points):
    """numba加速的三角形包围盒计算"""
    min_vals = np.array([
        min(points[0, 0], points[1, 0], points[2, 0]),
        min(points[0, 1], points[1, 1], points[2, 1]),
        min(points[0, 2], points[1, 2], points[2, 2])
    ])
    max_vals = np.array([
        max(points[0, 0], points[1, 0], points[2, 0]),
        max(points[0, 1], points[1, 1], points[2, 1]),
        max(points[0, 2], points[1, 2], points[2, 2])
    ])
    return np.vstack((min_vals, max_vals))


@njit
def triangles_share_vertex_numba(face1, face2):
    """numba加速的共享顶点检测"""
    for i in range(3):
        for j in range(3):
            if face1[i] == face2[j]:
                return True
    return False


@njit
def edge_triangle_intersect_numba(v0, v1, t0, t1, t2):
    """numba加速的边-三角形相交检测"""
    epsilon = 1e-9

    # 计算三角形平面的法向量
    edge1 = t1 - t0
    edge2 = t2 - t0
    normal = np.cross(edge1, edge2)

    # 检查线段是否与平面平行
    edge_dir = v1 - v0
    denominator = np.dot(normal, edge_dir)

    if abs(denominator) < epsilon:
        return False

    # 计算交点参数
    d = np.dot(normal, t0)
    t = (d - np.dot(normal, v0)) / denominator

    # 检查t是否在线段范围内
    if t < 0 or t > 1:
        return False

    # 计算交点
    intersection = v0 + t * edge_dir

    # 检查交点是否在三角形内
    # 使用重心坐标
    v0_tri = intersection - t0
    dot00 = np.dot(edge2, edge2)
    dot01 = np.dot(edge2, edge1)
    dot02 = np.dot(edge2, v0_tri)
    dot11 = np.dot(edge1, edge1)
    dot12 = np.dot(edge1, v0_tri)

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) and (v >= 0) and (u + v <= 1)


@njit
def triangles_intersect_numba(tri1_points, tri2_points):
    """numba加速的三角形相交检测"""
    # 检查三角形1的边与三角形2的相交
    for i in range(3):
        v0 = tri1_points[i]
        v1 = tri1_points[(i + 1) % 3]
        if edge_triangle_intersect_numba(v0, v1, tri2_points[0], tri2_points[1], tri2_points[2]):
            return True

    # 检查三角形2的边与三角形1的相交
    for i in range(3):
        v0 = tri2_points[i]
        v1 = tri2_points[(i + 1) % 3]
        if edge_triangle_intersect_numba(v0, v1, tri1_points[0], tri1_points[1], tri1_points[2]):
            return True

    return False


class FastBoundingBox:
    def __init__(self, points):
        if points.shape[0] == 3:  # 单个三角形
            self.min_max = triangle_bbox_numba(points)
        else:  # 多个点
            self.min_max = np.array([
                np.min(points, axis=0),
                np.max(points, axis=0)
            ])

    def intersects_with(self, other):
        return bbox_intersects_numba(self.min_max, other.min_max)


class FastBVHNode:
    def __init__(self, vertices, faces, tri_indices, depth=0, max_depth=20):
        self.left = None
        self.right = None
        self.triangle_indices = None
        self.bounding_box = None
        self.is_leaf_node = False
        self._build_node(vertices, faces, tri_indices, depth, max_depth)

    def _build_node(self, vertices, faces, tri_indices, depth, max_depth):
        # 如果三角形数量很少或达到最大深度，创建叶子节点
        if len(tri_indices) <= 10 or depth >= max_depth:
            self.triangle_indices = tri_indices
            self.is_leaf_node = True
            # 计算包围盒
            all_points = []
            for idx in tri_indices:
                triangle_points = vertices[faces[idx]]
                all_points.extend(triangle_points)
            self.bounding_box = FastBoundingBox(np.array(all_points))
            return

        # 计算包围盒
        all_points = []
        for idx in tri_indices:
            triangle_points = vertices[faces[idx]]
            all_points.extend(triangle_points)
        self.bounding_box = FastBoundingBox(np.array(all_points))

        # 选择分割轴（最长的轴）
        extent = self.bounding_box.min_max[1] - self.bounding_box.min_max[0]
        split_axis = np.argmax(extent)

        # 计算每个三角形在分割轴上的中心点
        centers = []
        for idx in tri_indices:
            triangle_points = vertices[faces[idx]]
            center = np.mean(triangle_points[:, split_axis])
            centers.append((center, idx))

        # 按中心点排序
        centers.sort()

        # 分割
        mid = len(centers) // 2
        left_indices = [item[1] for item in centers[:mid]]
        right_indices = [item[1] for item in centers[mid:]]

        # 创建子节点
        if left_indices:
            self.left = FastBVHNode(vertices, faces, left_indices, depth + 1, max_depth)
        if right_indices:
            self.right = FastBVHNode(vertices, faces, right_indices, depth + 1, max_depth)


class FastCollisionDetector:
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.face_normals = None

    def _precompute_data(self, vertices, faces):
        """预计算一些数据以加速检测"""
        self.vertices = vertices.astype(np.float64)
        self.faces = faces.astype(np.int32)

    def has_collision(self, vertices, faces):
        """
        快速检查是否存在碰撞
        """
        print("开始快速碰撞检测...")
        start_time = time.time()

        # 预计算数据
        self._precompute_data(vertices, faces)

        # 早期退出：如果三角形数量很少，直接暴力检测
        if len(faces) < 100:
            return self._brute_force_check()

        # 构建BVH树
        print("构建BVH树...")
        tree_start = time.time()
        tri_indices = list(range(len(faces)))
        root = FastBVHNode(self.vertices, self.faces, tri_indices)
        print(f"BVH树构建时间: {time.time() - tree_start:.3f}秒")

        # 检查碰撞
        print("检查碰撞...")
        check_start = time.time()
        has_collision = self._check_collision_bvh(root)
        print(f"碰撞检查时间: {time.time() - check_start:.3f}秒")
        print(f"总时间: {time.time() - start_time:.3f}秒")

        return has_collision

    def _brute_force_check(self):
        """暴力检测，适用于小规模网格"""
        n_faces = len(self.faces)
        for i in range(n_faces):
            for j in range(i + 1, n_faces):
                # 检查是否共享顶点
                if triangles_share_vertex_numba(self.faces[i], self.faces[j]):
                    continue

                # 检查三角形相交
                tri1_points = self.vertices[self.faces[i]]
                tri2_points = self.vertices[self.faces[j]]

                if triangles_intersect_numba(tri1_points, tri2_points):
                    print(f"发现碰撞: 三角形 {i} 与三角形 {j}")
                    return True
        return False

    def _check_collision_bvh(self, root):
        """使用BVH进行碰撞检测"""
        # 收集所有叶子节点
        leaf_nodes = []
        self._collect_leaf_nodes(root, leaf_nodes)

        # 检查叶子节点之间的碰撞
        n_leaves = len(leaf_nodes)
        for i in range(n_leaves):
            for j in range(i + 1, n_leaves):
                # 检查包围盒是否相交
                if not leaf_nodes[i].bounding_box.intersects_with(leaf_nodes[j].bounding_box):
                    continue

                # 检查叶子节点内的三角形
                if self._check_leaf_collision(leaf_nodes[i], leaf_nodes[j]):
                    return True

        return False

    def _collect_leaf_nodes(self, node, leaf_nodes):
        """收集所有叶子节点"""
        if node.is_leaf_node:
            leaf_nodes.append(node)
        else:
            if node.left:
                self._collect_leaf_nodes(node.left, leaf_nodes)
            if node.right:
                self._collect_leaf_nodes(node.right, leaf_nodes)

    def _check_leaf_collision(self, leaf1, leaf2):
        """检查两个叶子节点之间的碰撞"""
        for tri1_idx in leaf1.triangle_indices:
            for tri2_idx in leaf2.triangle_indices:
                # 检查是否共享顶点
                if triangles_share_vertex_numba(self.faces[tri1_idx], self.faces[tri2_idx]):
                    continue

                # 检查三角形相交
                tri1_points = self.vertices[self.faces[tri1_idx]]
                tri2_points = self.vertices[self.faces[tri2_idx]]

                if triangles_intersect_numba(tri1_points, tri2_points):
                    print(f"发现碰撞: 三角形 {tri1_idx} 与三角形 {tri2_idx}")
                    return True
        return False


def load_mesh(filename):
    """加载网格文件"""
    try:
        mesh = trimesh.load(filename)
        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            return mesh.vertices, mesh.faces
        else:
            raise ValueError("加载的文件不是有效的三角形网格")
    except Exception as e:
        print(f"加载网格文件失败: {e}")
        return None, None


def create_large_test_mesh():
    """创建一个大规模测试网格"""
    print("创建大规模测试网格...")

    # 创建一个球体网格
    sphere = trimesh.creation.icosphere(subdivisions=4)  # 这会创建大约2500个三角形

    # 复制并稍微移动以创建相交
    vertices1 = sphere.vertices
    faces1 = sphere.faces

    vertices2 = sphere.vertices + np.array([0.1, 0, 0])  # 稍微移动
    faces2 = sphere.faces + len(vertices1)  # 调整面索引

    # 合并两个球体
    all_vertices = np.vstack([vertices1, vertices2])
    all_faces = np.vstack([faces1, faces2])

    print(f"创建了 {len(all_vertices)} 个顶点, {len(all_faces)} 个三角形")
    return all_vertices, all_faces


def main():
    detector = FastCollisionDetector()

    # 测试大规模网格
    print("=== 大规模网格测试 ===")
    # vertices, faces = create_large_test_mesh()
    vertices, faces = load_mesh(sys.argv[1])
    has_collision = detector.has_collision(vertices, faces)
    print(f"结果: {'存在碰撞' if has_collision else '没有碰撞'}")


if __name__ == "__main__":
    main()
