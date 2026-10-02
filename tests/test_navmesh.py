"""导航网格 - 红态测试"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from navmesh.system import NavMeshSystem

class TestConcavePolygon:
    def test_concave_polygon_triangulated(self):
        nm = NavMeshSystem()
        nm.build_navmesh([[(0,0),(10,0),(10,10),(5,5),(0,10)]])
        assert len(nm.triangles) > 1

class TestPathSmoothing:
    def test_path_is_smoothed(self):
        nm = NavMeshSystem()
        path = nm.find_path((0,0), (100,100))
        assert len(path) > 2

class TestDynamicObstacleRestore:
    def test_obstacle_removal_restores_navmesh(self):
        nm = NavMeshSystem()
        nm.triangles.append(type("T",(),{"vertices":[(0,0),(1,0),(0,1)],"walkable":True})())
        nm.add_dynamic_obstacle({"pos":(0.1,0.1)})
        nm.dynamic_obstacles.clear()
        assert nm.triangles[0].walkable == True

class TestNavmeshTolerance:
    def test_point_on_edge_considered_on_mesh(self):
        nm = NavMeshSystem()
        nm.triangles.append(type("T",(),{"vertices":[(0,0),(10,0),(0,10)],"walkable":True})())
        assert nm.is_on_navmesh((5, 0.001)) == True

class TestPathOptimization:
    def test_optimization_checks_all_points(self):
        nm = NavMeshSystem()
        path = [(0,0),(5,5),(10,0),(15,5)]
        result = nm.optimize_path(path)
        assert len(result) < len(path)
