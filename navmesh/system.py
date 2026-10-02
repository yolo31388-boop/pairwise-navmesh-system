"""导航网格系统"""
import heapq
import math
from dataclasses import dataclass

EPS = 1e-9
ON_MESH_TOLERANCE = 1e-2
MAX_MERGED_VERTICES = 8
DEFAULT_OBSTACLE_RADIUS = 0.5
LOS_SAMPLE_STEP = 0.25
OBSTACLE_DETOUR_MARGIN = 0.5
FALLBACK_WAYPOINT_STEP = 10.0


@dataclass
class Triangle:
    tid: str
    vertices: list
    walkable: bool = True


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _signed_area(vertices):
    area = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _centroid(vertices):
    n = len(vertices)
    return (sum(v[0] for v in vertices) / n, sum(v[1] for v in vertices) / n)


def _point_segment_distance(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < EPS:
        return _dist(p, a)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return _dist(p, (ax + t * dx, ay + t * dy))


def _point_in_triangle_strict(p, tri):
    d1 = _cross(tri[0], tri[1], p)
    d2 = _cross(tri[1], tri[2], p)
    d3 = _cross(tri[2], tri[0], p)
    has_neg = (d1 < -EPS) or (d2 < -EPS) or (d3 < -EPS)
    has_pos = (d1 > EPS) or (d2 > EPS) or (d3 > EPS)
    return not (has_neg and has_pos)


def _point_in_polygon(p, vertices, tol=ON_MESH_TOLERANCE):
    """带容差的点在多边形内判定：边界附近（<=tol）也算在网格上。"""
    n = len(vertices)
    for i in range(n):
        if _point_segment_distance(p, vertices[i], vertices[(i + 1) % n]) <= tol:
            return True
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if (yi > p[1]) != (yj > p[1]):
            x_cross = (xj - xi) * (p[1] - yi) / (yj - yi) + xi
            if p[0] < x_cross:
                inside = not inside
        j = i
    return inside


def _triangulate(vertices):
    """耳切法三角化，支持凹多边形。"""
    n = len(vertices)
    if n < 3:
        return []
    if _signed_area(vertices) < 0:
        vertices = list(reversed(vertices))
    indices = list(range(n))
    triangles = []
    guard = 0
    while len(indices) > 3 and guard < n * n:
        guard += 1
        ear_found = False
        m = len(indices)
        for i in range(m):
            i0, i1, i2 = indices[i - 1], indices[i], indices[(i + 1) % m]
            a, b, c = vertices[i0], vertices[i1], vertices[i2]
            if _cross(a, b, c) <= EPS:
                continue
            if any(
                _point_in_triangle_strict(vertices[j], (a, b, c))
                for j in indices
                if j not in (i0, i1, i2)
            ):
                continue
            triangles.append([a, b, c])
            del indices[i]
            ear_found = True
            break
        if not ear_found:
            break
    if len(indices) == 3:
        triangles.append([vertices[i] for i in indices])
    return triangles


def _vertices_equal(a, b, tol=1e-6):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _try_merge(poly_a, poly_b):
    """若两个多边形共享一条边且合并后仍为凸多边形，返回合并后的顶点列表。"""
    shared = [v for v in poly_a if any(_vertices_equal(v, w) for w in poly_b)]
    if len(shared) != 2:
        return None

    def edge_endpoints(poly):
        idx = [i for i, v in enumerate(poly)
               if any(_vertices_equal(v, s) for s in shared)]
        if len(idx) != 2:
            return None
        i, j = idx
        if (i + 1) % len(poly) == j:
            return i, j
        if (j + 1) % len(poly) == i:
            return j, i
        return None

    ea = edge_endpoints(poly_a)
    eb = edge_endpoints(poly_b)
    if ea is None or eb is None:
        return None
    ia, ja = ea
    ib, jb = eb
    if not (_vertices_equal(poly_b[ib], poly_a[ja]) and
            _vertices_equal(poly_b[jb], poly_a[ia])):
        return None
    # 沿 a 反向从 s1 走到 s2，再沿 b 反向从 s2 走回 s1，去掉首尾重复
    chain_a = []
    k = ia
    while True:
        chain_a.append(poly_a[k])
        if k == ja:
            break
        k = (k - 1) % len(poly_a)
    chain_b = []
    k = ib
    while True:
        chain_b.append(poly_b[k])
        if k == jb:
            break
        k = (k - 1) % len(poly_b)
    merged = chain_a + chain_b[1:]
    cleaned = []
    for v in merged:
        if not cleaned or not _vertices_equal(v, cleaned[-1]):
            cleaned.append(v)
    if len(cleaned) > 1 and _vertices_equal(cleaned[0], cleaned[-1]):
        cleaned.pop()
    if len(cleaned) != len({(round(v[0], 6), round(v[1], 6)) for v in cleaned}):
        return None
    if len(cleaned) < 3 or len(cleaned) > MAX_MERGED_VERTICES or not _is_convex(cleaned):
        return None
    return cleaned


def _is_convex(vertices):
    n = len(vertices)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        cross = _cross(vertices[i], vertices[(i + 1) % n], vertices[(i + 2) % n])
        if abs(cross) < EPS:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _merge_polygons(polygons):
    """迭代合并共边且并集为凸的多边形，减少网格数量。"""
    polys = [list(p) for p in polygons]
    changed = True
    while changed:
        changed = False
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                merged = _try_merge(polys[i], polys[j])
                if merged is not None:
                    polys[i] = merged
                    del polys[j]
                    changed = True
                    break
            if changed:
                break
    return polys


class NavMeshSystem:
    def __init__(self):
        self.triangles: list = []
        self.dynamic_obstacles: list = []
        self._adjacency: dict = {}
        self._blocked_cache = None

    # ---------------- 构建 ----------------

    def build_navmesh(self, polygons: list) -> None:
        """耳切法三角化（支持凹多边形），并合并共边三角形为凸多边形。"""
        self.triangles = []
        self._adjacency = {}
        self._blocked_cache = None
        pieces = []
        for poly in polygons:
            pieces.extend(_triangulate(list(poly)))
        for vertices in _merge_polygons(pieces):
            tid = "t%d" % len(self.triangles)
            self.triangles.append(Triangle(tid, vertices))
        self._rebuild_adjacency()

    def _rebuild_adjacency(self):
        self._adjacency = {t.tid: [] for t in self.triangles}
        for i, a in enumerate(self.triangles):
            for b in self.triangles[i + 1:]:
                shared = sum(
                    1 for v in a.vertices
                    if any(_vertices_equal(v, w) for w in b.vertices)
                )
                if shared >= 2:
                    self._adjacency[a.tid].append(b.tid)
                    self._adjacency[b.tid].append(a.tid)

    # ---------------- 动态障碍 ----------------

    def add_dynamic_obstacle(self, obstacle: dict) -> None:
        """局部更新：只标记受影响的三角形，不重建网格；障碍移除后自动恢复。"""
        obstacle = dict(obstacle)
        obstacle.setdefault("radius", DEFAULT_OBSTACLE_RADIUS)
        self.dynamic_obstacles.append(obstacle)
        self._blocked_cache = None

    def remove_dynamic_obstacle(self, obstacle: dict) -> None:
        if obstacle in self.dynamic_obstacles:
            self.dynamic_obstacles.remove(obstacle)
            self._blocked_cache = None

    def _blocked_tids(self):
        # 缓存以 (id, len) 为键，外部 clear()/增删都会失效
        key = (id(self.dynamic_obstacles), len(self.dynamic_obstacles))
        if self._blocked_cache is not None and self._blocked_cache[0] == key:
            return self._blocked_cache[1]
        blocked = set()
        for obstacle in self.dynamic_obstacles:
            pos = obstacle["pos"]
            radius = obstacle.get("radius", DEFAULT_OBSTACLE_RADIUS)
            for tri in self.triangles:
                if self._tri_near_point(tri, pos, radius):
                    blocked.add(getattr(tri, "tid", None))
        self._blocked_cache = (key, blocked)
        return blocked

    @staticmethod
    def _tri_near_point(tri, point, radius):
        if _point_in_polygon(point, tri.vertices, tol=0.0):
            return True
        n = len(tri.vertices)
        return any(
            _point_segment_distance(point, tri.vertices[i], tri.vertices[(i + 1) % n]) <= radius
            for i in range(n)
        )

    def _tri_walkable(self, tri):
        return getattr(tri, "walkable", True) and getattr(tri, "tid", None) not in self._blocked_tids()

    # ---------------- 查询 ----------------

    def is_on_navmesh(self, point: tuple) -> bool:
        """带容差判断点是否在（可行走的）导航网格上，动态障碍覆盖处不可走。"""
        if self._point_in_obstacle(point):
            return False
        return any(
            getattr(tri, "walkable", True) and _point_in_polygon(point, tri.vertices)
            for tri in self.triangles
        )

    def _point_in_obstacle(self, point):
        return any(
            _dist(point, ob["pos"]) <= ob.get("radius", DEFAULT_OBSTACLE_RADIUS)
            for ob in self.dynamic_obstacles
        )

    def project_to_navmesh(self, point: tuple) -> tuple:
        """把点投影到最近的导航网格位置；不在网格上时返回最近边界点。"""
        if self.is_on_navmesh(point):
            return point
        best = None
        best_dist = float("inf")
        for tri in self.triangles:
            if not self._tri_walkable(tri):
                continue
            n = len(tri.vertices)
            for i in range(n):
                a, b = tri.vertices[i], tri.vertices[(i + 1) % n]
                dx, dy = b[0] - a[0], b[1] - a[1]
                length_sq = dx * dx + dy * dy
                t = 0.0 if length_sq < EPS else max(
                    0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_sq)
                )
                candidate = (a[0] + t * dx, a[1] + t * dy)
                d = _dist(point, candidate)
                if d < best_dist:
                    best_dist = d
                    best = candidate
        return best if best is not None else point

    def _find_triangle(self, point):
        if self._point_in_obstacle(point):
            return None
        for tri in self.triangles:
            if getattr(tri, "walkable", True) and _point_in_polygon(point, tri.vertices):
                return tri
        return None

    # ---------------- 寻路 ----------------

    def find_path(self, start: tuple, end: tuple) -> list:
        """A* 搜索三角形邻接图（欧氏距离启发，可采纳），再做视线拉直平滑。"""
        if not self.triangles:
            return self._interpolated_path(start, end)
        start = self.project_to_navmesh(start)
        end = self.project_to_navmesh(end)
        start_tri = self._find_triangle(start)
        end_tri = self._find_triangle(end)
        if start_tri is None or end_tri is None:
            return self._interpolated_path(start, end)
        corridor = self._astar(start_tri, end_tri)
        if corridor is None:
            return []
        raw = [start]
        raw += [_centroid(t.vertices) for t in corridor[1:-1]]
        raw.append(end)
        return self.optimize_path(raw)

    def _astar(self, start_tri, goal_tri):
        goal_center = _centroid(goal_tri.vertices)
        open_heap = [(_dist(_centroid(start_tri.vertices), goal_center), 0.0, start_tri.tid)]
        g_score = {start_tri.tid: 0.0}
        came_from = {}
        by_tid = {t.tid: t for t in self.triangles}
        while open_heap:
            _, g, tid = heapq.heappop(open_heap)
            if tid == goal_tri.tid:
                corridor = [by_tid[tid]]
                while tid in came_from:
                    tid = came_from[tid]
                    corridor.append(by_tid[tid])
                return corridor[::-1]
            if g > g_score.get(tid, float("inf")):
                continue
            tri = by_tid[tid]
            for nb_tid in self._adjacency.get(tid, []):
                nb = by_tid[nb_tid]
                if not self._tri_walkable(nb):
                    continue
                step = _dist(_centroid(tri.vertices), _centroid(nb.vertices))
                new_g = g + step
                if new_g < g_score.get(nb_tid, float("inf")):
                    g_score[nb_tid] = new_g
                    came_from[nb_tid] = tid
                    h = _dist(_centroid(nb.vertices), goal_center)
                    heapq.heappush(open_heap, (new_g + h, new_g, nb_tid))
        return None

    # ---------------- 路径优化 ----------------

    def optimize_path(self, path: list) -> list:
        """全点对视线拉直：先为穿障线段插入绕障点，再让每个点直连最远可见后续点。"""
        if len(path) <= 1:
            return list(path)
        expanded = [path[0]]
        for nxt in path[1:]:
            expanded.extend(self._detour(expanded[-1], nxt, 0))
        result = [expanded[0]]
        i = 0
        while i < len(expanded) - 1:
            j = len(expanded) - 1
            while j > i + 1 and not self._segment_walkable(expanded[i], expanded[j]):
                j -= 1
            result.append(expanded[j])
            i = j
        return result

    def _detour(self, a, b, depth):
        """线段被圆形动态障碍挡住时，在两侧绕行点中选总长度较短的一侧（递归处理多障碍）。"""
        if depth > 6 or self._segment_walkable(a, b):
            return [b]
        obstacle = self._first_hitting_obstacle(a, b)
        if obstacle is None:
            return [b]
        ox, oy = obstacle["pos"]
        radius = obstacle.get("radius", DEFAULT_OBSTACLE_RADIUS)
        if _dist(a, (ox, oy)) <= radius or _dist(b, (ox, oy)) <= radius:
            return [b]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        best = None
        best_len = float("inf")
        for side in (1.0, -1.0):
            flank = (
                ox + nx * side * (radius + OBSTACLE_DETOUR_MARGIN),
                oy + ny * side * (radius + OBSTACLE_DETOUR_MARGIN),
            )
            candidate = self._detour(a, flank, depth + 1) + self._detour(flank, b, depth + 1)
            total = _dist(a, candidate[0]) + sum(
                _dist(candidate[k], candidate[k + 1]) for k in range(len(candidate) - 1)
            )
            if total < best_len:
                best_len = total
                best = candidate
        return best if best is not None else [b]

    def _first_hitting_obstacle(self, a, b):
        for obstacle in self.dynamic_obstacles:
            radius = obstacle.get("radius", DEFAULT_OBSTACLE_RADIUS)
            if _point_segment_distance(obstacle["pos"], a, b) <= radius:
                return obstacle
        return None

    def _segment_walkable(self, a, b):
        """沿线段采样所有中间点，确认全程在可行走网格上且未被动态障碍阻挡。"""
        if self._segment_hits_obstacle(a, b):
            return False
        if not self.triangles:
            return True
        length = _dist(a, b)
        steps = max(2, int(math.ceil(length / LOS_SAMPLE_STEP)) + 1)
        for k in range(steps + 1):
            t = k / steps
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if not self.is_on_navmesh(p):
                return False
        return True

    def _segment_hits_obstacle(self, a, b):
        return any(
            _point_segment_distance(ob["pos"], a, b) <= ob.get("radius", DEFAULT_OBSTACLE_RADIUS)
            for ob in self.dynamic_obstacles
        )

    @staticmethod
    def _interpolated_path(start, end):
        length = _dist(start, end)
        steps = max(1, int(math.ceil(length / FALLBACK_WAYPOINT_STEP)))
        return [
            (start[0] + (end[0] - start[0]) * k / steps,
             start[1] + (end[1] - start[1]) * k / steps)
            for k in range(steps + 1)
        ]

    def _point_in_tri(self, point, tri):
        return _point_in_polygon(point, tri.vertices)
