import carla
import cv2
import numpy as np
import queue
from ultralytics import YOLO

# ================= 설정 =================
MODEL_PATH = "yolo11n.engine"  
IMAGE_W, IMAGE_H = 800, 600
FOV = 90

def main():
    # 1. 모델 로드
    print(f"[INFO] Loading YOLO Engine: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH, task='detect')
    print("[INFO] YOLO Loaded! Ready.")

    # 2. CARLA 연결
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 3. 트래픽 매니저 & 자율주행 차 생성
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    settings = world.get_settings()
    settings.synchronous_mode = True 
    settings.fixed_delta_seconds = 0.05 
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('cybertruck')[0]
    spawn_points = world.get_map().get_spawn_points()
    
    if not spawn_points:
        print("Spawn points not found!")
        return

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    vehicle.set_autopilot(True, tm.get_port())

    # 4. 카메라 설치 (★ 1인칭 시점으로 변경됨)
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(IMAGE_W))
    cam_bp.set_attribute('image_size_y', str(IMAGE_H))
    cam_bp.set_attribute('fov', str(FOV))
    
    # [수정된 부분] 운전석 대시보드 위치
    # z=1.4 → 1.7: 의자를 높여서 내려다보는 느낌으로 변경.
    # x=0.6 → 1.0: 앞유리 쪽으로 조금 더 붙어서 불필요한 하단 가림막 제거.
    # x=1.0 (앞쪽), z=2.0 (눈높이)
    camera_init_trans = carla.Transform(carla.Location(x=1.0, z=2.0), carla.Rotation(pitch=0))
    camera = world.spawn_actor(cam_bp, camera_init_trans, attach_to=vehicle)

    # 큐 생성
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    print("[INFO] Start 1st-Person Demo... (Press 'q' to exit)")
    
    try:
        while True:
            # 시간 1틱 진행
            world.tick()

            # 데이터 가져오기
            image = image_queue.get()
            
            # 이미지 변환
            img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            img_array = np.reshape(img_array, (IMAGE_H, IMAGE_W, 4))
            img_bgr = img_array[:, :, :3]

            # YOLO 추론 (동기)
            results = model.predict(img_bgr, verbose=False, half=True)
            annotated_frame = results[0].plot()

            # 화면 출력
            cv2.imshow("1st-Person Sync View", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 정리
        settings.synchronous_mode = False
        world.apply_settings(settings)
        vehicle.destroy()
        camera.destroy()
        cv2.destroyAllWindows()
        print("[INFO] Done.")

if __name__ == '__main__':
    main()