"""导航网格系统"""
import heapq
import math
from dataclasses import dataclass

EPSILON = 1e-6


@dataclass
class Triangle:
    tid: str
    vertices: list
    walkable: bool = True

    @property
    def center(self):
        n = len(self.vertices)
        return (sum(v[0] for v in self.vertices) / n,
                sum(v[1] for v in self.vertices) / n)


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _signed_area(pts):
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _point_in_triangle(p, verts, eps=EPSILON):
    """带容差的点在三角形内判断（含边界），支持任意顶点方向。"""
    a, b, c = verts[0], verts[1], verts[2]
    d1 = _cross(a, b, p)
    d2 = _cross(b, c, p)
    d3 = _cross(c, a, p)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def _strictly_in_triangle(p, a, b, c, eps=EPSILON):
    """CCW 三角形内的严格判断（耳切法用）。"""
    return (_cross(a, b, p) > eps and _cross(b, c, p) > eps
            and _cross(c, a, p) > eps)


def _triangulate(vertices):
    """耳切法三角化，正确处理凹多边形。"""
    pts = [tuple(v) for v in vertices]
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [pts]
    if _signed_area(pts) < 0:
        pts.reverse()
    indices = list(range(n))
    triangles = []
    guard = 0
    while len(indices) > 3 and guard < n * n:
        guard += 1
        ear_found = False
        m = len(indices)
        for i in range(m):
            i_prev = indices[(i - 1) % m]
            i_curr = indices[i]
            i_next = indices[(i + 1) % m]
            a, b, c = pts[i_prev], pts[i_curr], pts[i_next]
            if _cross(a, b, c) <= EPSILON:
                continue  # 凹顶点或共线，不是耳
            if any(_strictly_in_triangle(pts[j], a, b, c)
                   for j in indices if j not in (i_prev, i_curr, i_next)):
                continue
            triangles.append([a, b, c])
            indices.pop(i)
            ear_found = True
            break
        if not ear_found:
            break
    if len(indices) == 3:
        triangles.append([pts[indices[0]], pts[indices[1]], pts[indices[2]]])
    return triangles


def _closest_point_on_segment(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < EPSILON:
        return a
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return (ax + t * dx, ay + t * dy)


def _closest_point_on_triangle(p, verts):
    if _point_in_triangle(p, verts):
        return tuple(p)
    best, best_d = None, float("inf")
    for i in range(len(verts)):
        q = _closest_point_on_segment(p, verts[i], verts[(i + 1) % len(verts)])
        d = _dist(p, q)
        if d < best_d:
            best, best_d = q, d
    return best


class NavMeshSystem:
    def __init__(self):
        self.triangles: list = []
        self.dynamic_obstacles: list = []
        self._adjacency: dict = {}

    # ---------- 构建 ----------

    def build_navmesh(self, polygons: list) -> None:
        """耳切法三角化（支持凹多边形），并建立共边邻接表。"""
        self.triangles = []
        self._adjacency = {}
        tid = 0
        for poly in polygons:
            for verts in _triangulate(poly):
                self.triangles.append(Triangle(f"t{tid}", verts))
                tid += 1
        self._build_adjacency()

    def _build_adjacency(self):
        self._adjacency = {t.tid: [] for t in self.triangles}
        edge_map = {}
        for tri in self.triangles:
            verts = tri.vertices
            for i in range(len(verts)):
                a = tuple(round(c, 6) for c in verts[i])
                b = tuple(round(c, 6) for c in verts[(i + 1) % len(verts)])
                edge_map.setdefault(frozenset((a, b)), []).append(tri.tid)
        for tids in edge_map.values():
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    self._adjacency[tids[i]].append(tids[j])
                    self._adjacency[tids[j]].append(tids[i])

    # ---------- 寻路 ----------

    def find_path(self, start: tuple, end: tuple) -> list:
        """A* 搜索三角形通道，再拉直平滑；起终点不在网格上时先投影。"""
        start = (float(start[0]), float(start[1]))
        end = (float(end[0]), float(end[1]))
        if not self.triangles:
            return self._sample_line(start, end)
        start_tri = self._find_walkable_triangle(start)
        if start_tri is None:
            projected = self.project_to_navmesh(start)
            start_tri = self._find_walkable_triangle(projected)
            if start_tri is not None:
                start = projected
        end_tri = self._find_walkable_triangle(end)
        if end_tri is None:
            projected = self.project_to_navmesh(end)
            end_tri = self._find_walkable_triangle(projected)
            if end_tri is not None:
                end = projected
        if start_tri is None or end_tri is None:
            return self._sample_line(start, end)
        corridor = self._astar(start_tri, end_tri)
        if corridor is None:
            return [start, end]
        waypoints = [start] + [t.center for t in corridor[1:-1]] + [end]
        return self.optimize_path(waypoints)

    def _astar(self, start_tri, goal_tri):
        """三角形邻接图上的 A*，启发函数为到目标三角形中心的欧氏距离。"""
        tri_by_id = {t.tid: t for t in self.triangles}
        goal_center = goal_tri.center
        open_heap = [(0.0, start_tri.tid)]
        came_from = {}
        g_score = {start_tri.tid: 0.0}
        closed = set()
        while open_heap:
            _, current_id = heapq.heappop(open_heap)
            if current_id in closed:
                continue
            if current_id == goal_tri.tid:
                corridor = [tri_by_id[current_id]]
                while current_id in came_from:
                    current_id = came_from[current_id]
                    corridor.append(tri_by_id[current_id])
                corridor.reverse()
                return corridor
            closed.add(current_id)
            current = tri_by_id[current_id]
            for nb_id in self._adjacency.get(current_id, ()):
                neighbor = tri_by_id[nb_id]
                if not self.is_triangle_walkable(neighbor):
                    continue
                tentative = g_score[current_id] + _dist(current.center, neighbor.center)
                if tentative < g_score.get(nb_id, float("inf")):
                    g_score[nb_id] = tentative
                    came_from[nb_id] = current_id
                    heuristic = _dist(neighbor.center, goal_center)
                    heapq.heappush(open_heap, (tentative + heuristic, nb_id))
        return None

    def _sample_line(self, start, end, step=10.0):
        """无网格可用时的兜底：沿直线采样中间点。"""
        distance = _dist(start, end)
        n = max(2, int(distance / step) + 1)
        return [(start[0] + (end[0] - start[0]) * i / n,
                 start[1] + (end[1] - start[1]) * i / n)
                for i in range(n + 1)]

    # ---------- 动态障碍 ----------

    def add_dynamic_obstacle(self, obstacle: dict) -> None:
        """局部标记受影响的三角形，不重建网格；障碍移除后自动恢复。"""
        obstacle["affected"] = [
            id(tri) for tri in self.triangles
            if self._obstacle_hits_triangle(obstacle, tri)
        ]
        self.dynamic_obstacles.append(obstacle)

    def remove_dynamic_obstacle(self, obstacle: dict) -> None:
        if obstacle in self.dynamic_obstacles:
            self.dynamic_obstacles.remove(obstacle)

    def is_triangle_walkable(self, tri) -> bool:
        if not getattr(tri, "walkable", True):
            return False
        tri_id = id(tri)
        return not any(tri_id in obs.get("affected", ())
                       for obs in self.dynamic_obstacles)

    def _obstacle_hits_triangle(self, obstacle, tri) -> bool:
        pos = obstacle.get("pos")
        if pos is None:
            return False
        verts = getattr(tri, "vertices", None)
        if not verts:
            return False
        radius = obstacle.get("radius", 0.0)
        if radius <= 0:
            return _point_in_triangle(pos, verts)
        if _point_in_triangle(pos, verts):
            return True
        return any(
            _dist(pos, _closest_point_on_segment(pos, verts[i], verts[(i + 1) % len(verts)])) <= radius + EPSILON
            for i in range(len(verts))
        )

    # ---------- 查询 ----------

    def is_on_navmesh(self, point: tuple, eps: float = 1e-6) -> bool:
        """带浮点容差的点在网格上判断。"""
        return any(_point_in_triangle(point, tri.vertices, eps)
                   for tri in self.triangles)

    def project_to_navmesh(self, point: tuple) -> tuple:
        """把不在网格上的点投影到最近的可行走三角形上。"""
        best, best_d = tuple(point), float("inf")
        for tri in self.triangles:
            if not self.is_triangle_walkable(tri):
                continue
            candidate = _closest_point_on_triangle(point, tri.vertices)
            d = _dist(point, candidate)
            if d < best_d:
                best, best_d = candidate, d
        return best

    # ---------- 路径优化 ----------

    def optimize_path(self, path: list) -> list:
        """拉直路径：对每一对非相邻点做全程可见性检查，实时考虑动态障碍。"""
        if len(path) <= 2:
            return list(path)
        result = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._segment_walkable(path[i], path[j]):
                j -= 1
            result.append(path[j])
            i = j
        return result

    def _segment_walkable(self, a, b) -> bool:
        """沿线段采样所有中间点，逐点检查是否落在可行走网格上。"""
        if not self.triangles:
            return True
        steps = max(2, int(_dist(a, b) / 0.5) + 1)
        for i in range(steps + 1):
            t = i / steps
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if not self._point_walkable(p):
                return False
        return True

    def _point_walkable(self, p) -> bool:
        return any(_point_in_triangle(p, tri.vertices)
                   and self.is_triangle_walkable(tri)
                   for tri in self.triangles)

    def _find_walkable_triangle(self, point):
        for tri in self.triangles:
            if (_point_in_triangle(point, tri.vertices)
                    and self.is_triangle_walkable(tri)):
                return tri
        return None

    def _point_in_tri(self, point, tri):
        return _point_in_triangle(point, tri.vertices)
