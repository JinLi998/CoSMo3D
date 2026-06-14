"""Transfer point-cloud segmentation labels to mesh faces and export colored GLB."""

from collections import Counter, defaultdict

import matplotlib.cm as cm
import numpy as np
import trimesh

FACE_BATCH_SIZE = 1000
MAX_FILL_ITER = 50

DEFAULT_COLOR_MAP = {
    0: (128, 128, 128),
    1: (255, 0, 0),
    2: (255, 255, 0),
    3: (0, 0, 255),
    4: (0, 255, 0),
    5: (0, 255, 255),
    6: (255, 0, 255),
    7: (128, 128, 128),
    8: (255, 165, 0),
    9: (139, 69, 19),
    10: (240, 230, 140),
}


def build_face_adjacency(model):
    vertex_to_faces = defaultdict(list)
    for face_idx, face in enumerate(model.faces):
        for vertex in face:
            vertex_to_faces[vertex].append(face_idx)

    adjacency = [[] for _ in range(len(model.faces))]
    for face_idx, face in enumerate(model.faces):
        neighbor_candidates = set()
        for vertex in face:
            neighbor_candidates.update(vertex_to_faces[vertex])
        for neighbor in neighbor_candidates:
            if neighbor != face_idx and len(set(face) & set(model.faces[neighbor])) >= 2:
                adjacency[face_idx].append(neighbor)
    return adjacency


def preprocess_faces(model):
    faces = model.faces
    vertices = np.asarray(model.vertices, dtype=np.float64)
    num_faces = len(faces)

    face_bboxes = np.zeros((num_faces, 2, 3), dtype=np.float64)
    face_planes = np.zeros((num_faces, 4), dtype=np.float64)
    face_vertices = np.zeros((num_faces, 3, 3), dtype=np.float64)
    face_centers = np.zeros((num_faces, 3), dtype=np.float64)

    for i in range(num_faces):
        v = vertices[faces[i]]
        face_vertices[i] = v
        face_bboxes[i, 0] = v.min(axis=0)
        face_bboxes[i, 1] = v.max(axis=0)
        face_centers[i] = v.mean(axis=0)

        v0, v1, v2 = v
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        d = -np.dot(normal, v0)
        face_planes[i] = [normal[0], normal[1], normal[2], d]

    return face_vertices, face_bboxes, face_planes, face_centers


def vectorized_point_to_face_distance(points, face_vertices, face_planes):
    a, b, c, d = face_planes
    v0, v1, v2 = face_vertices

    plane_dist = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = points - v0

    dot00 = np.sum(v0v2 * v0v2)
    dot01 = np.sum(v0v2 * v0v1)
    dot02 = np.sum(v0p * v0v2, axis=1)
    dot11 = np.sum(v0v1 * v0v1)
    dot12 = np.sum(v0p * v0v1, axis=1)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-12)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    in_triangle = (u >= 0) & (v >= 0) & (u + v <= 1)

    dist_to_v0 = np.linalg.norm(points - v0, axis=1)
    dist_to_v1 = np.linalg.norm(points - v1, axis=1)
    dist_to_v2 = np.linalg.norm(points - v2, axis=1)
    vertex_dist = np.minimum(np.minimum(dist_to_v0, dist_to_v1), dist_to_v2)

    return np.where(in_triangle, plane_dist, vertex_dist)


def assign_face_labels(model, points, point_labels):
    face_vertices, face_bboxes, face_planes, face_centers = preprocess_faces(model)
    num_points = len(points)
    num_faces = len(model.faces)

    closest_face_indices = np.zeros(num_points, dtype=int)
    min_distances = np.full(num_points, np.inf, dtype=np.float64)

    num_batches = (num_faces + FACE_BATCH_SIZE - 1) // FACE_BATCH_SIZE
    for batch_idx in range(num_batches):
        start = batch_idx * FACE_BATCH_SIZE
        end = min((batch_idx + 1) * FACE_BATCH_SIZE, num_faces)

        for face_idx in range(start, end):
            bbox_min, bbox_max = face_bboxes[face_idx]
            in_bbox = np.all((points >= bbox_min - 1e-3) & (points <= bbox_max + 1e-3), axis=1)
            if not np.any(in_bbox):
                continue

            points_in_bbox = points[in_bbox]
            distances = vectorized_point_to_face_distance(
                points_in_bbox, face_vertices[face_idx], face_planes[face_idx]
            )

            bbox_point_indices = np.where(in_bbox)[0]
            for i in range(len(points_in_bbox)):
                global_idx = bbox_point_indices[i]
                if distances[i] < min_distances[global_idx]:
                    min_distances[global_idx] = distances[i]
                    closest_face_indices[global_idx] = face_idx

    face_point_labels = defaultdict(list)
    for point_idx, face_idx in enumerate(closest_face_indices):
        face_point_labels[face_idx].append(int(point_labels[point_idx]))

    face_labels = np.full(num_faces, -1, dtype=int)
    for face_idx, labels in face_point_labels.items():
        if labels:
            face_labels[face_idx] = Counter(labels).most_common(1)[0][0]

    return face_labels, face_centers


def fill_unlabeled_faces(face_labels, adjacency, face_centers):
    for _ in range(MAX_FILL_ITER):
        new_labels = face_labels.copy()
        unlabeled_faces = np.where(face_labels == -1)[0]
        labeled_faces = np.where(face_labels != -1)[0]
        if len(labeled_faces) == 0:
            break

        for face_idx in unlabeled_faces:
            neighbor_labels = [face_labels[n] for n in adjacency[face_idx] if face_labels[n] != -1]
            if neighbor_labels:
                new_labels[face_idx] = Counter(neighbor_labels).most_common(1)[0][0]

        remaining_unlabeled = np.where(new_labels == -1)[0]
        if len(remaining_unlabeled) > 0:
            for face_idx in remaining_unlabeled:
                dists = np.linalg.norm(face_centers[labeled_faces] - face_centers[face_idx], axis=1)
                closest_idx = labeled_faces[np.argmin(dists)]
                new_labels[face_idx] = face_labels[closest_idx]

        face_labels = new_labels
        if np.sum(face_labels == -1) == 0:
            break

    face_labels[face_labels == -1] = 0
    return face_labels


def label_to_face_colors(face_labels, color_map=None):
    color_map = color_map or DEFAULT_COLOR_MAP
    unique_labels = np.unique(face_labels)
    if color_map is None or len(color_map) < len(unique_labels):
        hues = np.linspace(0, 1, len(unique_labels), endpoint=False)
        np.random.shuffle(hues)
        dynamic_map = {
            int(label): (np.array(cm.hsv(hue)[:3]) * 255).astype(np.uint8)
            for label, hue in zip(unique_labels, hues)
        }
        color_map = {**DEFAULT_COLOR_MAP, **dynamic_map}

    face_colors = np.array(
        [color_map.get(int(label), DEFAULT_COLOR_MAP[0]) for label in face_labels],
        dtype=np.uint8,
    )
    face_colors = np.hstack([face_colors, np.full((len(face_colors), 1), 255, dtype=np.uint8)])
    return face_colors, color_map


def load_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from {mesh_path}: {type(mesh)}")
    return mesh


def align_mesh_to_pointcloud(mesh, angle=np.pi / 2, direction=(1, 0, 0)):
    """Rotate mesh to the point-cloud frame used by 3DCoMPaT samples."""
    aligned = mesh.copy()
    rotation = trimesh.transformations.rotation_matrix(angle=angle, direction=direction)
    aligned.apply_transform(rotation)
    return aligned


def transfer_labels_to_mesh(mesh, points, point_labels, align_to_points=True):
    target_mesh = align_mesh_to_pointcloud(mesh) if align_to_points else mesh
    adjacency = build_face_adjacency(target_mesh)
    face_labels, face_centers = assign_face_labels(target_mesh, points, point_labels)
    face_labels = fill_unlabeled_faces(face_labels, adjacency, face_centers)
    return face_labels


def export_colored_glb(mesh, face_labels, output_path, color_map=None):
    face_colors, used_color_map = label_to_face_colors(face_labels, color_map=color_map)
    colored_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        face_colors=face_colors,
        process=False,
    )
    colored_mesh.export(output_path, file_type="glb")
    return used_color_map
