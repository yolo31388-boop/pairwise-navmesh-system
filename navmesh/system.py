"""导航网格 - 含5个bug"""
from dataclasses import dataclass, field

@dataclass
class Triangle:
    tid: str
    vertices: list
    walkable: bool = True

class NavMeshSystem:
    def __init__(self):
        self.triangles: list = []
        self.dynamic_obstacles: list = []

    def build_navmesh(self, polygons: list) -> None:
        # bug1: 不处理凹多边形
        for poly in polygons:
            self.triangles.append(Triangle("t", poly))

    def find_path(self, start: tuple, end: tuple) -> list:
        # bug2: 启发函数不合适，路径不平滑
        return [start, end]

    def add_dynamic_obstacle(self, obstacle: dict) -> None:
        # bug3: 直接移除三角形，不恢复
        self.dynamic_obstacles.append(obstacle)
        for tri in self.triangles:
            if self._point_in_tri(obstacle["pos"], tri):
                tri.walkable = False

    def is_on_navmesh(self, point: tuple) -> bool:
        # bug4: 精确相等
        for tri in self.triangles:
            if point in tri.vertices:
                return True
        return False

    def optimize_path(self, path: list) -> list:
        # bug5: 只检查相邻两点
        return path

    def _point_in_tri(self, point, tri):
        return False
