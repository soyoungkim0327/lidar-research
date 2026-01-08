import carla
import cv2
import numpy as np
import open3d as o3d
import queue
from ultralytics import YOLO

# ================= 설정 =================
MODEL_PATH = "yolo11n.engine"  
IMAGE_W, IMAGE_H = 800, 600
FOV = 90

def main():
    # 1. 모델 로드 (엔진)
    print(f"[INFO] Loading YOLO Engine: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH, task='detect')
    print("[INFO] YOLO Loaded! Ready for Sync Demo.")

    # 2. CARLA 연결
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # 맵이 Town04가 아니면 로드 (선택사항)
    # client.load_world('Town04') 

    # 3. 트래픽 매니저 & 자율주행 차 생성
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True) # TM도 동기화

    settings = world.get_settings()
    settings.synchronous_mode = True # ★ 핵심: 서버 시간을 멈춤
    settings.fixed_delta_seconds = 0.05 # 20 FPS 고정
    world.apply_settings(settings)

    # 차 소환
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('cybertruck')[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    vehicle.set_autopilot(True, tm.get_port()) # 자율주행 ON

    # 4. 카메라 설치
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(IMAGE_W))
    cam_bp.set_attribute('image_size_y', str(IMAGE_H))
    cam_bp.set_attribute('fov', str(FOV))
    
    # 차 뒤에서 찍는 구도 (3인칭)
    camera_init_trans = carla.Transform(carla.Location(x=-6, z=2.5), carla.Rotation(pitch=-15))
    camera = world.spawn_actor(cam_bp, camera_init_trans, attach_to=vehicle)

    # 큐(Queue) 생성 - 데이터를 받아낼 바구니
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    print("[INFO] Start Sync Simulation... (Press 'q' to exit)")
    
    try:
        while True:
            # (1) 시간을 1프레임 진행시킴 (틱!)
            world.tick()

            # (2) 카메라 데이터 가져오기 (시간을 멈췄으므로 무조건 있음)
            image = image_queue.get()
            
            # (3) 이미지 변환 (Raw -> BGR)
            img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            img_array = np.reshape(img_array, (IMAGE_H, IMAGE_W, 4))
            img_bgr = img_array[:, :, :3]

            # (4) YOLO 추론 (동기 방식: 다 될 때까지 기다림)
            # engine을 썼기 때문에 여기서 기다려도 순식간에 끝남!
            results = model.predict(img_bgr, verbose=False, half=True)
            
            # (5) 결과 그리기
            annotated_frame = results[0].plot()

            # (6) 화면 출력
            cv2.imshow("Sync Demo (Perfect Accuracy)", annotated_frame)

            # 종료 키
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