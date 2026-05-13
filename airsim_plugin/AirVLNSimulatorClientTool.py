from collections import deque
import multiprocessing
import msgpackrpc
import time
import airsim
import threading
import random
import copy
import numpy as np
import cv2
import math
import os,sys

import tqdm

cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")

from src.common.runtime_config import runtime_config_from_env
from utils.logger import logger


CAMERA_LOCAL_POSES = {
    '0': ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    '1': ((0.0, -1.0, 0.0), (0.0, 0.0, -90.0)),
    '2': ((0.0, 1.0, 0.0), (0.0, 0.0, 90.0)),
    '3': ((0.0, 0.0, 0.0), (-90.0, 0.0, 0.0)),
}


def _pose_from_value(value) -> airsim.Pose:
    if isinstance(value, airsim.Pose):
        return value
    if isinstance(value, (list, tuple)) and len(value) >= 7:
        return airsim.Pose(
            position_val=airsim.Vector3r(value[0], value[1], value[2]),
            orientation_val=airsim.Quaternionr(value[3], value[4], value[5], value[6]),
        )
    raise TypeError(f"Unsupported pose value: {type(value)}")


def _position_array(position) -> np.ndarray:
    return np.array([position.x_val, position.y_val, position.z_val], dtype=np.float64)


def _quaternion_array(quaternion) -> np.ndarray:
    values = np.array(
        [quaternion.x_val, quaternion.y_val, quaternion.z_val, quaternion.w_val],
        dtype=np.float64,
    )
    norm = np.linalg.norm(values)
    if norm <= 0:
        raise ValueError("zero-norm quaternion")
    return values / norm


def _quaternion_multiply(q1, q2) -> airsim.Quaternionr:
    x1, y1, z1, w1 = _quaternion_array(q1)
    x2, y2, z2, w2 = _quaternion_array(q2)
    return airsim.Quaternionr(
        x_val=w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        y_val=w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        z_val=w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w_val=w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _rotate_vector(quaternion, vector) -> np.ndarray:
    q = _quaternion_array(quaternion)
    x, y, z, w = q
    q_vec = np.array([x, y, z], dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    return v + 2.0 * np.cross(q_vec, np.cross(q_vec, v) + w * v)


def _quaternion_angle_degrees(q1, q2) -> float:
    dot = abs(float(np.dot(_quaternion_array(q1), _quaternion_array(q2))))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def _pose_errors(expected_pose: airsim.Pose, actual_position, actual_orientation):
    position_error = float(
        np.linalg.norm(_position_array(expected_pose.position) - _position_array(actual_position))
    )
    orientation_error = _quaternion_angle_degrees(expected_pose.orientation, actual_orientation)
    return position_error, orientation_error


def _expected_camera_pose(vehicle_pose: airsim.Pose, camera_name: str) -> airsim.Pose:
    offset, euler_degrees = CAMERA_LOCAL_POSES.get(str(camera_name), CAMERA_LOCAL_POSES['0'])
    camera_position = _position_array(vehicle_pose.position) + _rotate_vector(vehicle_pose.orientation, offset)
    pitch, roll, yaw = (math.radians(v) for v in euler_degrees)
    camera_orientation = _quaternion_multiply(vehicle_pose.orientation, airsim.to_quaternion(pitch, roll, yaw))
    return airsim.Pose(
        position_val=airsim.Vector3r(*camera_position.tolist()),
        orientation_val=camera_orientation,
    )


class BaseSensor:
    def __init__(self) -> None:
        pass

    def retrieve(self):
        raise NotImplementedError()

class State(BaseSensor):
    def __init__(self, client, drone_name=''):
        self.data = {'position': None, 'linear_velocity': None, 'linear_acceleration':None,
                     'orientation':None, 'angular_velocity':None, 'angular_acceleration':None}
        self.client: airsim.MultirotorClient = client
        self.drone_name = drone_name

    def retrieve(self):
        data = self.client.getMultirotorState(vehicle_name=self.drone_name)
        collision = {}
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
        collision['has_collided'] = collision_info.has_collided
        collision['object_name'] = data.collision.object_name
        gps_location = [data.gps_location.latitude,data.gps_location.longitude,data.gps_location.altitude]
        timestamp = data.timestamp
        position = list(data.kinematics_estimated.position)
        linear_velocity = list(data.kinematics_estimated.linear_velocity)
        linear_acceleration = list(data.kinematics_estimated.linear_acceleration)
        orientation = list(data.kinematics_estimated.orientation)
        angular_velocity = list(data.kinematics_estimated.angular_velocity)
        angular_acceleration = list(data.kinematics_estimated.angular_acceleration)

        self.data.update({'collision': collision, 
                          'gps_location': gps_location,
                          'timestamp': timestamp, 
                          'position': position,
                          'linear_velocity': linear_velocity,
                          'linear_acceleration': linear_acceleration,
                          'orientation': orientation,
                          'angular_velocity': angular_velocity,
                          'angular_acceleration': angular_acceleration
                          })
        return self.data
        
        
class Imu(BaseSensor):
    def __init__(self, client, drone_name='', imu_name=''):
        self.data = {}
        self.client: airsim.MultirotorClient = client
        self.drone_name = drone_name
        self.imu_name = imu_name

    def retrieve(self):
        data = self.client.getImuData(imu_name=self.imu_name,vehicle_name=self.drone_name)
        time_stamp = data.time_stamp
        orientation = data.orientation
        angular_velocity = list(data.angular_velocity)
        linear_acceleration = list(data.linear_acceleration)
        q0, q1, q2, q3 = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        rotation_matrix = np.array(([1-2*(q2*q2+q3*q3),2*(q1*q2-q3*q0),2*(q1*q3+q2*q0)],
                                      [2*(q1*q2+q3*q0),1-2*(q1*q1+q3*q3),2*(q2*q3-q1*q0)],
                                      [2*(q1*q3-q2*q0),2*(q2*q3+q1*q0),1-2*(q1*q1+q2*q2)])).tolist()
        self.data.update({'time_stamp': time_stamp, 'rotation': rotation_matrix, 'orientation': list(data.orientation),
                          'linear_acceleration': linear_acceleration, 'angular_velocity': angular_velocity})
        return self.data


class MyThread(threading.Thread):
    def __init__(self, func, args, kwargs=None):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.flag_ok = False

    def run(self):
        self.result = self.func(*self.args, **self.kwargs)
        self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None


class AirVLNSimulatorClientTool:
    def __init__(self, machines_info) -> None:
        self.machines_info = copy.deepcopy(machines_info)
        self.socket_clients = []
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in machines_info ]
        self.airsim_client_endpoints = [[None for _ in list(item['open_scenes'])] for item in machines_info ]
        self.airsim_ports = []
        self.airsim_ip = '127.0.0.1'
        self._init_check()
        self.objects_name_cnt = [[0 for _ in list(item['open_scenes'])] for item in machines_info ]

    def _init_check(self) -> None:
        ips = [item['MACHINE_IP'] for item in self.machines_info]
        assert len(ips) == len(set(ips)), 'MACHINE_IP repeat'

    def _confirmSocketConnection(self, socket_client: msgpackrpc.Client) -> bool:
        try:
            socket_client.call('ping')
            print("Connected\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            return True
        except:
            try:
                print("Ping returned false\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            except:
                print('Ping returned false')
            return False

    def _waitForAirSimReady(
        self,
        airsim_client: airsim.MultirotorClient,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 1.0,
        label: str,
    ) -> bool:
        start_time = time.monotonic()
        last_log_time = -poll_interval_seconds
        attempts = 0
        last_error = None

        while True:
            attempts += 1
            elapsed = time.monotonic() - start_time
            try:
                airsim_client.confirmConnection()
                vehicles = airsim_client.listVehicles()
                vehicle_name = vehicles[0] if vehicles else ''
                airsim_client.enableApiControl(True, vehicle_name=vehicle_name)
                airsim_client.armDisarm(True, vehicle_name=vehicle_name)
                airsim_client.takeoffAsync(vehicle_name=vehicle_name).join()
                state = airsim_client.getMultirotorState(vehicle_name=vehicle_name)
                if state is None:
                    raise RuntimeError("getMultirotorState returned None")
                image_datas = airsim_client.simGetImages(
                    [airsim.ImageRequest('0', airsim.ImageType.Scene, pixels_as_float=False, compress=True)]
                )
                if not image_datas:
                    raise RuntimeError("simGetImages returned no responses")
                image_data = image_datas[0]
                if image_data.width <= 0 or image_data.height <= 0 or len(image_data.image_data_uint8) == 0:
                    raise RuntimeError(
                        f"invalid image response width={image_data.width} height={image_data.height}"
                    )
                logger.info(
                    f"AirSim ready {label}: elapsed={time.monotonic() - start_time:.2f}s attempts={attempts}"
                )
                return True
            except Exception as e:
                last_error = e
                elapsed = time.monotonic() - start_time
                if elapsed - last_log_time >= 5.0 or attempts == 1:
                    logger.info(
                        f"waiting for AirSim {label}: elapsed={elapsed:.2f}s "
                        f"timeout={timeout_seconds:.2f}s error={e}"
                    )
                    last_log_time = elapsed
                if elapsed >= timeout_seconds:
                    logger.error(
                        f"AirSim readiness timeout {label}: elapsed={elapsed:.2f}s "
                        f"attempts={attempts} last_error={last_error}"
                    )
                    return False
                sleep_seconds = min(poll_interval_seconds, max(0.0, timeout_seconds - elapsed))
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

    def _confirmConnection(self, *, timeout_seconds: float) -> bool:
        threads = []
        thread_results = []
        for index_1, _ in enumerate(self.airsim_clients):
            threads.append([])
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                airsim_client = self.airsim_clients[index_1][index_2]
                if airsim_client is None:
                    threads[index_1].append(None)
                    continue
                label = f"machine={index_1} scene={index_2}"
                threads[index_1].append(
                    MyThread(
                        self._waitForAirSimReady,
                        (
                            airsim_client,
                        ),
                        {
                            "timeout_seconds": timeout_seconds,
                            "label": label,
                        },
                    )
                )
        for index_1, _ in enumerate(threads):
            for thread in threads[index_1]:
                if thread is not None:
                    thread.setDaemon(True)
                    thread.start()
        for index_1, _ in enumerate(threads):
            for thread in threads[index_1]:
                if thread is not None:
                    thread.join()
        for index_1, _ in enumerate(threads):
            for thread in threads[index_1]:
                if thread is not None:
                    thread_results.append(bool(thread.get_result()) and thread.flag_ok)

        return bool(thread_results) and (np.array(thread_results) == True).all()

    def _closeSocketConnection(self) -> None:
        socket_clients = self.socket_clients

        for socket_client in socket_clients:
            try:
                socket_client.close()
            except Exception as e:
                pass

        self.socket_clients = []
        return

    def _closeConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    try:
                        self.airsim_clients[index_1][index_2].close()
                    except Exception as e:
                        pass

        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in self.machines_info]
        self.airsim_client_endpoints = [[None for _ in list(item['open_scenes'])] for item in self.machines_info]
        return

    def _resetAirSimClientTimeouts(self, *, timeout_value: float) -> None:
        for index_1, _ in enumerate(self.airsim_client_endpoints):
            for index_2, endpoint in enumerate(self.airsim_client_endpoints[index_1]):
                if endpoint is None:
                    self.airsim_clients[index_1][index_2] = None
                    continue
                ip, port = endpoint
                self.airsim_clients[index_1][index_2] = airsim.MultirotorClient(
                    ip=ip, port=port, timeout_value=timeout_value
                )

    def run_call(self, airsim_timeout: int=300) -> None:
        socket_clients = []
        for index, item in enumerate(self.machines_info):
            socket_clients.append(
                msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=300)
            )

        for socket_client in socket_clients:
            if not self._confirmSocketConnection(socket_client):
                logger.error('cannot establish socket')
                raise Exception('cannot establish socket')

        self.socket_clients = socket_clients


        before = time.time()
        self._closeConnection()

        def _run_command(index, socket_client: msgpackrpc.Client):
            logger.info(f'开始打开场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
            logger.info(f'gpus: {self.machines_info[index]}')
            result = socket_client.call('reopen_scenes', socket_client.address._host, list(zip(self.machines_info[index]['open_scenes'], self.machines_info[index]['gpus'])))
            print(list(zip(self.machines_info[index]['open_scenes'], self.machines_info[index]['gpus'])))
            if result[0] == False:
                logger.error(f'打开场景失败，机器: {socket_client.address._host}:{socket_client.address._port}')
                raise Exception('打开场景失败')
            
            assert len(result[1]) == 2, '打开场景失败'
            print('waiting for airsim connection...')
            default_boot_seconds = 3 * len(self.machines_info[index]['open_scenes']) + 15
            boot_seconds = runtime_config_from_env(
                scene_boot_seconds_default=default_boot_seconds
            ).scene_boot_seconds
            ip = result[1][0]
            ports = result[1][1]
            if isinstance(ip, bytes):
                ip = ip.decode('utf-8')
            self.airsim_ip = ip
            self.airsim_ports = ports
            assert str(ip) == str(socket_client.address._host), '打开场景失败'
            assert len(ports) == len(self.machines_info[index]['open_scenes']), '打开场景失败'
            for i, port in enumerate(ports):
                if self.machines_info[index]['open_scenes'][i] is None:
                    self.airsim_clients[index][i] = None
                    self.airsim_client_endpoints[index][i] = None
                else:  
                    readiness_rpc_timeout = min(5.0, max(1.0, float(boot_seconds)))
                    self.airsim_clients[index][i] = airsim.MultirotorClient(
                        ip=ip, port=port, timeout_value=readiness_rpc_timeout
                    )
                    self.airsim_client_endpoints[index][i] = (ip, port)

            logger.info(f'打开场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
            return boot_seconds

        threads = []
        thread_results = []
        for index, socket_client in enumerate(socket_clients):
            threads.append(
                MyThread(_run_command, (index, socket_client))
            )
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        command_results = []
        for thread in threads:
            command_results.append(thread.get_result())
            thread_results.append(thread.flag_ok)
        threads = []
        
        if not (np.array(thread_results) == True).all():
            raise Exception('打开场景失败')

        readiness_timeout_values = [float(result) for result in command_results if result is not None]
        if not readiness_timeout_values:
            raise Exception('打开场景失败')
        readiness_timeout_seconds = max(readiness_timeout_values)
        assert self._confirmConnection(timeout_seconds=readiness_timeout_seconds), 'server connect failed'
        self._resetAirSimClientTimeouts(timeout_value=airsim_timeout)

        after = time.time()
        diff = after - before
        logger.info(f"启动时间：{diff}")

        self._closeSocketConnection()
    
    def closeScenes(self):
        try:
            socket_clients = []
            for index, item in enumerate(self.machines_info):
                socket_clients.append(
                    msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=300)
                )

            for socket_client in socket_clients:
                if not self._confirmSocketConnection(socket_client):
                    logger.error('cannot establish socket')
                    raise Exception('cannot establish socket')

            self.socket_clients = socket_clients

            self._closeConnection()

            def _run_command(index, socket_client: msgpackrpc.Client):
                logger.info(f'开始关闭所有场景，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
                result = socket_client.call('close_scenes', socket_client.address._host)
                logger.info(f'关闭所有场景完毕，机器{index}: {socket_client.address._host}:{socket_client.address._port}')
                return

            threads = []
            for index, socket_client in enumerate(socket_clients):
                threads.append(
                    MyThread(_run_command, (index, socket_client))
                )
            for thread in threads:
                thread.setDaemon(True)
                thread.start()
            for thread in threads:
                thread.join()
            threads = []

            self._closeSocketConnection()
        except Exception as e:
            logger.error(e)



    def move_to_next_pose(self, poses_list: list, fly_types: list):
        def _move(airsim_client: airsim.VehicleClient, pose: airsim.Pose, fly_type: str):
            if airsim_client is None:
                raise Exception('error')
                return
            
            vehicles = airsim_client.listVehicles()
            vehicle_name = vehicles[0] if vehicles else ''
            state_sensor = State(airsim_client, drone_name=vehicle_name)
            imu_sensor = Imu(airsim_client, drone_name=vehicle_name, imu_name="Imu")
            airsim_client.simPause(False)
            future = None
            action_timeout_sec = runtime_config_from_env().action_timeout_seconds

            if fly_type == 'move':
                drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom
                future = airsim_client.moveToPositionAsync(
                    pose.position.x_val,
                    pose.position.y_val,
                    pose.position.z_val,
                    velocity=1,
                    timeout_sec=action_timeout_sec,
                    drivetrain=drivetrain,
                    vehicle_name=vehicle_name,
                )

            elif fly_type == 'rotate':
                (pitch, roll, yaw) = airsim.to_eularian_angles(pose.orientation)
                future = airsim_client.rotateToYawAsync(
                    math.degrees(yaw),
                    timeout_sec=action_timeout_sec,
                    vehicle_name=vehicle_name,
                )

            if future is not None:
                future.join()
            else:
                time.sleep(0.05)
            airsim_client.simPause(True)
            
            state_info = copy.deepcopy(state_sensor.retrieve())
            imu_info = copy.deepcopy(imu_sensor.retrieve())
            position = np.array(state_info['position'])

            results = []
            collision = False
            if state_info['collision']['has_collided']:
                collision = True
            results.append({'sensors':{'state':state_info,'imu':imu_info}})
            return {'states': results,'collision':collision}


        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_move, (self.airsim_clients[index_1][index_2], poses_list[index_1][index_2], fly_types[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        result_poses_list = []
        error_flag = False
        for index_1, _ in enumerate(threads):
            result_poses_list.append([])
            for index_2, _ in enumerate(threads[index_1]):
                result =  threads[index_1][index_2].get_result()
                result_poses_list[index_1].append(
                    result
                )
                if result is None:
                    error_flag = True
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('move by position failed.')
            return None
        if error_flag:
            return None
        return result_poses_list
    
    def _waitForVehiclePose(self, airsim_client, pose, vehicle_name: str, runtime_config, label: str):
        timeout_seconds = max(0.0, runtime_config.set_pose_verify_timeout_seconds)
        poll_interval_seconds = max(0.001, runtime_config.set_pose_poll_interval_seconds)
        position_tolerance = runtime_config.set_pose_position_tolerance_m
        orientation_tolerance = runtime_config.set_pose_orientation_tolerance_deg
        start_time = time.monotonic()
        attempts = 0
        last_error = None

        while True:
            attempts += 1
            try:
                state = airsim_client.getMultirotorState(vehicle_name=vehicle_name)
                position_error, orientation_error = _pose_errors(
                    pose,
                    state.kinematics_estimated.position,
                    state.kinematics_estimated.orientation,
                )
                if position_error <= position_tolerance and orientation_error <= orientation_tolerance:
                    if runtime_config.verbose_pose:
                        logger.info(
                            f"pose verified {label}: pos_error={position_error:.4f}m "
                            f"ori_error={orientation_error:.3f}deg attempts={attempts} "
                            f"elapsed={time.monotonic() - start_time:.3f}s"
                        )
                    return state, position_error, orientation_error
                last_error = (
                    f"pos_error={position_error:.4f}m>{position_tolerance:.4f}m "
                    f"ori_error={orientation_error:.3f}deg>{orientation_tolerance:.3f}deg"
                )
            except Exception as e:
                last_error = e

            elapsed = time.monotonic() - start_time
            if elapsed >= timeout_seconds:
                raise RuntimeError(
                    f"pose verification timeout {label}: elapsed={elapsed:.3f}s "
                    f"attempts={attempts} last_error={last_error}"
                )
            airsim_client.simPause(False)
            time.sleep(min(poll_interval_seconds, max(0.0, timeout_seconds - elapsed)))
            airsim_client.simPause(True)

    def setPoses(self, poses: list) -> bool:
        def _setPoses(airsim_client: airsim.VehicleClient, pose: airsim.Pose, ) -> None:
            if airsim_client is None:
                raise Exception('error')
                return
            pose = _pose_from_value(pose)
            vehicles = airsim_client.listVehicles()
            vehicle_name = vehicles[0] if vehicles else ''
            runtime_config = runtime_config_from_env()
            verbose_pose = runtime_config.verbose_pose
            if verbose_pose:
                print(f"set pose vehicle={vehicle_name} target={pose.position}", flush=True)
            settle_seconds = runtime_config.set_pose_settle_seconds
            airsim_client.simPause(True)
            airsim_client.simSetVehiclePose(pose=pose, ignore_collision=True, vehicle_name=vehicle_name)
            if vehicle_name:
                airsim_client.simSetObjectScale(vehicle_name, airsim.Vector3r(0.5, 0.5, 0.5))
            if settle_seconds > 0:
                airsim_client.simPause(False)
                time.sleep(settle_seconds)
            airsim_client.simPause(True)
            state, position_error, orientation_error = self._waitForVehiclePose(
                airsim_client,
                pose,
                vehicle_name,
                runtime_config,
                label=f"vehicle={vehicle_name}",
            )
            if verbose_pose:
                print(
                    f"set pose done vehicle={vehicle_name} state={state.kinematics_estimated.position} "
                    f"pos_error={position_error:.4f} ori_error={orientation_error:.3f}",
                    flush=True,
                )
            return

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                #print("index_1,index_2: ", str(index_1),str(index_2))
                threads[index_1].append(
                    MyThread(_setPoses, (self.airsim_clients[index_1][index_2], poses[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('setPoses失败')
            return False

        return True
    
    def _validateImageResponses(
        self,
        image_datas,
        active_cameras,
        *,
        rgb_only: bool,
        target_pose,
        runtime_config,
    ) -> None:
        expected_response_count = len(active_cameras) if rgb_only else 2 * len(active_cameras)
        if len(image_datas) < expected_response_count:
            raise RuntimeError(
                f"simGetImages returned {len(image_datas)} responses, expected {expected_response_count}"
            )

        vehicle_pose = _pose_from_value(target_pose) if target_pose is not None else None
        for idx, camera_name in enumerate(active_cameras):
            rgb_resp = image_datas[idx] if rgb_only else image_datas[2 * idx]
            if rgb_resp.width <= 0 or rgb_resp.height <= 0 or len(rgb_resp.image_data_uint8) == 0:
                raise RuntimeError(
                    f"invalid RGB image camera={camera_name} width={rgb_resp.width} height={rgb_resp.height}"
                )
            if vehicle_pose is None:
                continue
            if not hasattr(rgb_resp, "camera_position") or not hasattr(rgb_resp, "camera_orientation"):
                raise RuntimeError("AirSim image response does not expose camera pose")
            expected_camera_pose = _expected_camera_pose(vehicle_pose, str(camera_name))
            position_error, orientation_error = _pose_errors(
                expected_camera_pose,
                rgb_resp.camera_position,
                rgb_resp.camera_orientation,
            )
            if (
                position_error > runtime_config.image_pose_tolerance_m
                or orientation_error > runtime_config.image_orientation_tolerance_deg
            ):
                raise RuntimeError(
                    f"stale or mismatched image camera={camera_name}: "
                    f"pos_error={position_error:.4f}m>{runtime_config.image_pose_tolerance_m:.4f}m "
                    f"ori_error={orientation_error:.3f}deg>"
                    f"{runtime_config.image_orientation_tolerance_deg:.3f}deg"
                )

    def getImageResponses(self, cameras=('0', '1', '2', '3'), poses=None):
        runtime_config = runtime_config_from_env()
        active_cameras = ('0',) if runtime_config.front_view_only else tuple(cameras)

        def _getImages(airsim_client: airsim.VehicleClient, target_pose=None):
            if airsim_client is None:
                raise Exception('client is None.')
                return None, None
            rgb_only = runtime_config.rgb_only
            rgb_compress = runtime_config.rgb_compress
            image_verify_timeout = max(0.0, runtime_config.image_verify_timeout_seconds)
            image_poll_interval = max(0.001, runtime_config.image_poll_interval_seconds)
            start_time = time.monotonic()
            attempts = 0
            last_error = None
            while True:
                attempts += 1
                try:
                    ImageRequest = []
                    for camera_name in active_cameras:
                        ImageRequest.append(airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=rgb_compress))
                        if not rgb_only:
                            ImageRequest.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False))
                    image_settle_seconds = runtime_config.image_settle_seconds
                    if image_settle_seconds > 0:
                        airsim_client.simPause(False)
                        time.sleep(image_settle_seconds)
                    else:
                        airsim_client.simPause(True)
                    image_datas = airsim_client.simGetImages(requests=ImageRequest)
                    airsim_client.simPause(True)
                    self._validateImageResponses(
                        image_datas,
                        active_cameras,
                        rgb_only=rgb_only,
                        target_pose=target_pose,
                        runtime_config=runtime_config,
                    )
                    images, depth_images = [], []
                    for idx, camera_name in enumerate(active_cameras):
                        if rgb_only:
                            rgb_resp = image_datas[idx]
                        else:
                            rgb_resp = image_datas[2 * idx]
                        if rgb_compress:
                            image = rgb_resp.image_data_uint8
                        else:
                            image_array = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8)
                            pixel_count = int(rgb_resp.width) * int(rgb_resp.height)
                            if image_array.size == pixel_count * 4:
                                image_array = image_array.reshape(int(rgb_resp.height), int(rgb_resp.width), 4)[:, :, :3]
                            elif image_array.size == pixel_count * 3:
                                image_array = image_array.reshape(int(rgb_resp.height), int(rgb_resp.width), 3)
                            else:
                                raise ValueError(
                                    f"Unexpected raw RGB buffer size={image_array.size} "
                                    f"for {rgb_resp.width}x{rgb_resp.height}"
                                )
                            image = image_array.copy()
                        if rgb_only:
                            depth_image = None
                        else:
                            depth_resp = image_datas[2* idx + 1]
                            depth_img_in_meters = airsim.list_to_2d_float_array(depth_resp.image_data_float, depth_resp.width, depth_resp.height)
                            depth_image = (np.clip(depth_img_in_meters, 0, 100) / 100 * 255).astype(np.uint8)
                        images.append(image)
                        depth_images.append(depth_image)
                    break
                except Exception as e:
                    last_error = e
                    elapsed = time.monotonic() - start_time
                    logger.error(
                        f"图片获取错误 attempt={attempts} elapsed={elapsed:.3f}s "
                        f"timeout={image_verify_timeout:.3f}s: {e}"
                    )
                    if elapsed >= image_verify_timeout:
                        raise RuntimeError(
                            f"图片获取失败 after {elapsed:.3f}s attempts={attempts}: {last_error}"
                        )
                    airsim_client.simPause(False)
                    time.sleep(min(image_poll_interval, max(0.0, image_verify_timeout - elapsed)))
                    airsim_client.simPause(True)
            return images, depth_images

        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                target_pose = None
                if poses is not None:
                    target_pose = poses[index_1][index_2]
                threads[index_1].append(
                    MyThread(_getImages, (self.airsim_clients[index_1][index_2], target_pose))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponses失败')
            return None

        return responses

    def getSensorInfo(self, ):
        def get_sensor_info(airsim_client: airsim.VehicleClient, ):
            state_sensor = State(airsim_client, )
            imu_sensor = Imu(airsim_client)
            state_info = state_sensor.retrieve()
            imu_info = imu_sensor.retrieve()
            return {'sensors': {'state':state_info, 'imu': imu_info}}
        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(get_sensor_info, (self.airsim_clients[index_1][index_2], ))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()

        results = []
        for index_1, _ in enumerate(threads):
            results.append([])
            for index_2, _ in enumerate(threads[index_1]):
                results[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getSensorInfo failed.')
            return None
        return results
