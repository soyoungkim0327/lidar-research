import carla
import cv2
import numpy as np
import open3d as o3d
import queue
from ultralytics import YOLO
from matplotlib import cm

# ================= 화면 크기 설정 (2배 확대) =================
TOTAL_WIDTH = 1600
LIDAR_H = 500
YOLO_H = 900
LIDAR_W = int(TOTAL_WIDTH / 2)

TOP_MARGIN = 50
GAP = 40

# 기타 설정
LIDAR_RANGE = 80.0
MODEL_PATH = "yolo11n.engine"

# 색상 테이블
try:
    VIRIDIS = np.array(cm.get_cmap('plasma').colors)
except AttributeError:
    import matplotlib.pyplot as plt
    VIRIDIS = np.array(plt.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def main():
    print(f"[INFO] Loading YOLO Engine: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH, task='detect')
    print(f"[INFO] Ready. Safe Mode: Auto-Focus Enabled.")

    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    # ==============================================================================
    # [1] Visualizer 셋업
    # ==============================================================================
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='1. LiDAR 3D (Auto Focus)', 
                      width=LIDAR_W, height=LIDAR_H, 
                      left=0, top=TOP_MARGIN) 
    vis.get_render_option().background_color = np.asarray([0.0, 0.0, 0.0])
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().show_coordinate_frame = True

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='2. LiDAR BEV (Map View)', 
                       width=LIDAR_W, height=LIDAR_H, 
                       left=LIDAR_W, top=TOP_MARGIN) 
    vis2.get_render_option().background_color = np.asarray([0.0, 0.0, 0.0])
    vis2.get_render_option().point_size = 3.0
    vis2.get_render_option().show_coordinate_frame = False

    lidar_point_list = o3d.geometry.PointCloud()
    lidar_point_list.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    lidar_point_list.colors = o3d.utility.Vector3dVector(np.array([[1.0, 1.0, 1.0]]))
    vis.add_geometry(lidar_point_list)

    bev_point_list = o3d.geometry.PointCloud()
    bev_point_list.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    bev_point_list.colors = o3d.utility.Vector3dVector(np.array([[1.0, 1.0, 1.0]]))
    vis2.add_geometry(bev_point_list)

    first_view_reset = False

    # ==============================================================================
    # [2] Sync 모드 설정
    # ==============================================================================
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.05 
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('cybertruck')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    vehicle.set_autopilot(True, tm.get_port())

    # 센서 설치
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(TOTAL_WIDTH))
    cam_bp.set_attribute('image_size_y', str(YOLO_H))
    cam_bp.set_attribute('fov', '90')
    camera_init_trans = carla.Transform(carla.Location(x=-6, z=2.5), carla.Rotation(pitch=-15))
    camera = world.spawn_actor(cam_bp, camera_init_trans, attach_to=vehicle)

    lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', str(LIDAR_RANGE))
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '600000') 
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_init_trans = carla.Transform(carla.Location(x=0, z=2.0))
    lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=vehicle)

    image_queue = queue.Queue()
    lidar_queue = queue.Queue()
    camera.listen(image_queue.put)
    lidar.listen(lidar_queue.put)

    cv2.namedWindow("Main: YOLO (Sync)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Main: YOLO (Sync)", TOTAL_WIDTH, YOLO_H)
    cv2.moveWindow("Main: YOLO (Sync)", 0, TOP_MARGIN + LIDAR_H + GAP)

    print("[INFO] Loop Start.")
    
    try:
        while True:
            world.tick()
            image_data = image_queue.get()
            lidar_data = lidar_queue.get()
            
            # --- LiDAR ---
            data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
            data = np.reshape(data, (int(data.shape[0] / 4), 4))
            

            # [수정] 좌우 반전 해결 (Y축 반전)
            # CARLA (Left-Handed) -> Open3D (Right-Handed) 보정
            points = data[:, :-1]
            points[:, 1] = -points[:, 1]  # <--- ★ 핵심 수정: Y축(좌우) 반전
            
            intensity = data[:, -1]
            intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
            colors = np.c_[
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])
            ]
            bev_points_data = points.copy(); bev_points_data[:, 2] = 0

            lidar_point_list.points = o3d.utility.Vector3dVector(points)
            lidar_point_list.colors = o3d.utility.Vector3dVector(colors)
            bev_point_list.points = o3d.utility.Vector3dVector(bev_points_data)
            bev_point_list.colors = o3d.utility.Vector3dVector(colors)

            vis.update_geometry(lidar_point_list)
            vis2.update_geometry(bev_point_list)


            if not first_view_reset and points.shape[0] > 100:
                # 1. 3D 화면 (왼쪽): 운전석 시점 (뒤에서 앞을 봄)
                vis.reset_view_point(True)  # [필수] 점들이 있는 곳으로 카메라 이동 (깜깜함 방지)
                ctr = vis.get_view_control()
                ctr.set_lookat([0.0, 0.0, 0.0]) # 차 중심을 봄
                ctr.set_up([0.0, 0.0, 1.0])     # 머리 위는 하늘(Z)
                ctr.set_front([-1.0, 0.0, 0.5]) # ★ 차 뒤쪽(-1.0) 위(+0.5)에서 앞을 봄
                ctr.set_zoom(0.4)               # 적당히 줌 당김

                # 2. 2D 화면 (오른쪽): 네비게이션 시점
                vis2.reset_view_point(True)
                ctr2 = vis2.get_view_control()
                ctr2.set_lookat([0.0, 0.0, 0.0])
                ctr2.set_front([0.0, 0.0, 1.0]) # 하늘에서 수직으로 봄
                ctr2.set_up([1.0, 0.0, 0.0])    # ★ 차가 가는 방향(X)이 모니터 위쪽이 됨
                ctr2.set_zoom(0.45)
                
                first_view_reset = True


            vis.poll_events()
            vis.update_renderer()
            vis2.poll_events()
            vis2.update_renderer()

            # --- YOLO ---
            img_array = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
            img_array = np.reshape(img_array, (YOLO_H, TOTAL_WIDTH, 4))
            img_bgr = img_array[:, :, :3]
            results = model.predict(img_bgr, verbose=False, half=True)
            yolo_frame = results[0].plot()

            cv2.imshow("Main: YOLO (Sync)", yolo_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        vehicle.destroy()
        camera.destroy()
        lidar.destroy()
        vis.destroy_window()
        vis2.destroy_window()
        cv2.destroyAllWindows()
        print("[INFO] Done.")

if __name__ == '__main__':
    main()
