import json
import math
import ssl
import time
import traceback
import uuid
from haversine import haversine, inverse_haversine
from classes import Logic, MqttClient, GpsClient
from data.config import GpsConfig


class Agent():
    
    """

    An agent that can be used for the 2022 Arena map and the Integration map. It can perform two tasks, go-home and move-to. 
    The main use of this agent is to be able to change frequency of the Livespectr ogram in the QTSpectrogram file.
    This agent is based on the examples provided by WARA-PS at https://github.com/wara-ps/waraps-agent-examples.git . 
    For more information about how the agents work, checkout https://api.docs.waraps.org

    Attributes:
    spectrogram : LiveSpectrogram
        The spectrogram is a LiveSpectrogram from the file QtSpectrogram.  
    streamer : PyRtmpStreamer
        A streamer that starts the streaming to WARA PS

    
    """

    def __init__(self, spectrogram, streamer) -> None:
        self.gps = GpsClient()
        self.logic = Logic()

        # MQTT setup
        self.mqtt_client = MqttClient()
        self.mqtt_client.client.on_connect = self.on_connect
        self.mqtt_client.client.on_message = self.on_message
        self.mqtt_client.client.on_disconnect = self.on_disconnect
        self.connect(self.mqtt_client)

        self.spectrogram = spectrogram
        self.streamer = streamer

    def connect(self, client):
        """Connect to the broker using the mqtt client"""
        if client.tls_connection:
            client.client.username_pw_set(client.user, client.password)
            client.client.tls_set(cert_reqs=ssl.CERT_NONE)
            client.client.tls_insecure_set(True)
        try:
            client.client.connect(client.broker, client.port, 60)
            client.client.loop_start()
        except Exception as exc:
            print(f"Failed to connect to broker {client.broker}:{client.port}")
            print(exc)
            exit()

    def publish(self, topic, msg):
        """Publish the message (msg) on the given topic"""
        self.mqtt_client.client.publish(topic, msg)

    def disconnect(self, client):
        """Disconnect the client from the broker"""
        client.client.disconnect()
        client.client.loop_stop()

    def on_connect(self, client, userdata, flags, rc):
        """Callback triggered when the client connects to the broker"""
        try:
            if rc == 0:
                print(f"Connected to MQTT Broker: {self.mqtt_client.broker}:{self.mqtt_client.port}")
                self.mqtt_client.client.subscribe(self.mqtt_client.listen_topic)
                print(f"Subscribing to {self.mqtt_client.listen_topic}")
            else:
                print(f"Error to connect : {rc}")
        except Exception:
            print(traceback.format_exc())

    def on_message(self, client, userdata, msg):
        """Is triggered when a message is published on topics agent subscribes to"""
        try:
            msg_str = msg.payload.decode("utf-8")
            msg_json = json.loads(msg_str)

            if msg_json["command"] == "start-task":
                print("RECEIVED COMMAND 'start-task'")

                task_uuid = msg_json["task-uuid"]
                task = msg_json["task"]
                com_uuid = msg_json["com-uuid"]

                msg_res_json = {
                    "agent-uuid": self.logic.uuid,
                    "com-uuid": str(uuid.uuid4()),
                    "fail-reason": "",
                    "response": "",
                    "response-to": com_uuid,
                    "task-uuid": task_uuid
                }

                msg_feed_json = {
                    "agent-uuid": self.logic.uuid,
                    "com-uuid": str(uuid.uuid4()),
                    "status": "",
                    "task-uuid": task_uuid
                }

                if self.is_task_supported(task) and not self.logic.task_running:
                    # if task["name"] == "move-to":
                    #     # If we want to frequency using the WARA PS 2022 Arena we do not have the change-frequency 
                    #     # implemnted and therefore we "cheat" by using move-to instead. The altitude represents the frequency in MHz
                    #     self.logic.task_running = True
                    #     self.logic.task_running_uuid = task_uuid
                    #     msg_res_json["response"] = "running"
                    #     msg_res_json["fail-reason"] = ""
                    #     msg_feed_json["status"] = "running"

                    #     self.spectrogram.set_frequency(float(task["params"]["waypoint"]["altitude"]))


                    #     lat = task["params"]["waypoint"]["latitude"]
                    #     lon = task["params"]["waypoint"]["longitude"]
                    #     alt = task["params"]["waypoint"]["altitude"]

                    #     self.logic.task_target = (lat, lon, alt)

                    #     self.initialize_speed(task["params"]["speed"])

                        
                    # if task["name"] == "go-home":
                    #     # A way of implementing the functionality to stop stream in the WARA PS 2022 Arena.
                    #     self.logic.task_running = True
                    #     self.logic.task_running_uuid = task_uuid
                    #     msg_res_json["response"] = "running"
                    #     msg_res_json["fail-reason"] = ""
                    #     msg_feed_json["status"] = "running"

                    #     lat = GpsConfig.LATITUDE
                    #     lon = GpsConfig.LONGITUDE
                    #     alt = GpsConfig.ALTITUDE

                    #     self.logic.task_target = (lat, lon, alt)

                    #     self.initialize_speed(task["params"]["speed"])




                    #     self.streamer.stop_rtmp_stream()
                    #     #self.streamer.stop_local_stream()

        
                    if task["name"] == "change-frequency":
                        #use atlas
                        self.logic.task_running = True
                        self.logic.task_running_uuid = task_uuid
                        msg_res_json["response"] = "running"
                        msg_res_json["fail-reason"] = ""
                        msg_feed_json["status"] = "running"

                        self.spectrogram.set_frequency(task["params"]["frequency"])

                        lat = GpsConfig.LATITUDE
                        lon = GpsConfig.LONGITUDE
                        alt = GpsConfig.ALTITUDE

                        self.logic.task_target = (lat, lon, alt)

                    if task["name"] == "start-stream":
                        #use atlas
                        self.logic.task_running = True
                        self.logic.task_running_uuid = task_uuid
                        msg_res_json["response"] = "running"
                        msg_res_json["fail-reason"] = ""
                        msg_feed_json["status"] = "running"

                        self.streamer.start_rtmp_stream()

                        lat = GpsConfig.LATITUDE
                        lon = GpsConfig.LONGITUDE
                        alt = GpsConfig.ALTITUDE

                        self.logic.task_target = (lat, lon, alt)

                else:
                    if self.logic.task_running:  # Task running
                        msg_res_json["fail-reason"] = "A task is already running"
                    else:  # Task not supported
                        msg_res_json["fail-reason"] = "Task is not supported"
                    msg_res_json["response"] = "failed"
                    msg_feed_json["status"] = "failed"

                msg_res_str = json.dumps(msg_res_json)
                msg_feed_str = json.dumps(msg_feed_json)
                exec_topic: str = f"{self.mqtt_client.base_topic}/exec"
                self.mqtt_client.client.publish(f"{exec_topic}/response", msg_res_str)
                self.mqtt_client.client.publish(f"{exec_topic}/feedback", msg_feed_str)
                print(f"SENT RESPONSE! : {msg_res_str}")
                print(f"SENT FEEDBACK! : {msg_feed_str}")

            # Command that affects running task
            elif msg_json["command"] == "signal-task":
                print("RECEIVED COMMAND 'signal-task'")
                signal = msg_json["signal"]
                signal_task_uuid = msg_json["task-uuid"]
                com_uuid = msg_json["com-uuid"]

                msg_res_json = {
                    "com-uuid": str(uuid.uuid4()),
                    "response": "",
                    "response-to": com_uuid,
                    "task-uuid": self.logic.task_running_uuid
                }
                msg_feed_json = {
                    "agent-uuid": self.logic.uuid,
                    "com-uuid": str(uuid.uuid4()),
                    "status": "",
                    "task-uuid": self.logic.task_running_uuid
                }

                # Task signals
                if self.logic.task_running_uuid == signal_task_uuid:
                    if signal == "$abort":
                        msg_feed_json["status"] = "aborted"
                        self.reinstate_agent_variables()

                    elif signal == "$enough":
                        msg_feed_json["status"] = "enough"
                        self.reinstate_agent_variables()

                    elif signal == "$pause":
                        msg_feed_json["status"] = "paused"
                        self.logic.task_pause_flag = True

                    elif signal == "$continue":
                        msg_feed_json["status"] = "running"
                        self.logic.task_pause_flag = False

                    msg_res_json["response"] = "ok"
                else:
                    msg_res_json["fail-reason"] = "Invalid task-uuid"
                    msg_res_json["response"] = "failed"
                    msg_feed_json["status"] = "failed"

                msg_res_str = json.dumps(msg_res_json)
                msg_feed_str = json.dumps(msg_feed_json)
                exec_topic: str = f"{self.mqtt_client.base_topic}/exec"
                self.mqtt_client.client.publish(f"{exec_topic}/response", msg_res_str)
                self.mqtt_client.client.publish(f"{exec_topic}/feedback", msg_feed_str)
                print(f"SENT RESPONSE! : {msg_res_str}")
                print(f"SENT FEEDBACK! : {msg_feed_str}")

        except Exception:
            print(traceback.format_exc())

    def on_disconnect(self, client, userdata, rc):
        """Is triggered when the client gets disconnected from the broker"""
        print(f"Client Got Disconnected from the broker {userdata} with code {rc}")
        if rc == 5:
            print("No (or Wrong) Credentials, Edit in '.env'")

    def initialize_speed(self, speed: str) -> None:
        """Set current speed of agent according to string"""
        if speed == "slow":
            self.gps.speed = GpsConfig.SPEED_SLOW
        elif speed == "standard":
            self.gps.speed = GpsConfig.SPEED_STANDARD
        elif speed == "fast":
            self.gps.speed = GpsConfig.SPEED_FAST
        else:
            self.gps.speed = GpsConfig.SPEED_STANDARD

    def reinstate_agent_variables(self):
        """Reset agent's task variables"""
        self.logic.task_running_uuid = ""
        self.logic.task_running = False
        self.logic.task_pause_flag = False
        self.logic.task_target = ()
        self.logic.path = []

    def send_heartbeat(self):
        """Publish the heartbeat information of the agent to its topic"""
        json_msg = {
            "name": self.logic.name,
            "agent-type": self.logic.type,
            "agent-description": self.logic.description,
            "agent-uuid": self.logic.uuid,
            "levels": self.logic.level,
            "rate": self.logic.rate,
            "stamp": time.time(),
            "type": "HeartBeat"
        }
        str_msg = json.dumps(json_msg)
        self.publish(
            f"{self.mqtt_client.base_topic}/heartbeat", str_msg)

    def send_sensor_info(self):
        """Publish the sensor information of the agent to its topic"""
        json_msg = {
            "name": self.logic.name,
            "rate": self.logic.rate,
            "sensor-data-provided": [
                "position",
                "speed",
                "course",
                "heading",
            ],
            "stamp": time.time(),
            "type": "SensorInfo"
        }
        str_msg = json.dumps(json_msg)
        self.publish(
            f"{self.mqtt_client.base_topic}/sensor_info", str_msg)

    def send_position(self):
        """Publish the position of the agent to its topic"""
        json_msg = {
            "latitude": self.gps.lat,
            "longitude": self.gps.lon,
            "altitude": self.gps.alt,
            "type": "GeoPoint"
        }
        str_msg = json.dumps(json_msg)
        self.publish(
            f"{self.mqtt_client.base_topic}/sensor/position", str_msg)

    def send_speed(self):
        """Publish the speed of the agent to its topic"""
        speed = self.gps.speed
        self.publish(f"{self.mqtt_client.base_topic}/sensor/speed", speed)

    def send_course(self):
        """Publish the course of the agent to its topic"""
        course = self.gps.course
        self.publish(f"{self.mqtt_client.base_topic}/sensor/course", course)

    def send_heading(self):
        """Publish the heading of the agent to its topic"""
        heading = self.gps.heading
        self.publish(
            f"{self.mqtt_client.base_topic}/sensor/heading", heading)

    def send_direct_execution_info(self):
        """Publish the direct execution information of the agent to its topic"""
        json_msg = {
            "type": "DirectExecutionInfo",
            "name": self.logic.name,
            "rate": self.logic.rate,
            "stamp": time.time(),
            "tasks-available": self.logic.tasks_available
        }
        str_msg = json.dumps(json_msg)
        self.publish(f"{self.mqtt_client.base_topic}/direct_execution_info", str_msg)

    def set_speed(self, speed: float) -> None:
        """Set current speed according to float (m/s)"""
        self.gps.speed = speed

    def set_heading(self, heading: float) -> None:
        """Set the current heading"""
        self.gps.heading = heading

    def set_course(self, course: float) -> None:
        """Set the current course"""
        self.gps.course = course

    def move_to_target(self, current: tuple, target: tuple) -> None:
        """Move agent to the target position in Latitude, Longitude and Altitude"""
        *current_no_alt, _ = current
        *target_no_alt, _ = target

        # Check distance to target in kilometers
        distance = haversine(current_no_alt, target_no_alt)
        if distance <= 0.01:
            print("Reached target....", end=" ")
            self.gps.lat, self.gps.lon = target_no_alt
            if not self.logic.path:  # no path
                self.logic.task_running = False
                print("Task Complete")
            else:  # there is a path
                lat = self.logic.path[0]["latitude"]
                lon = self.logic.path[0]["longitude"]
                alt = self.logic.path[0]["altitude"]
                self.logic.task_target = (lat, lon, alt)
                self.logic.path.pop(0)
                print("Moving to next target")
            return

        bearing = self.bearing(current_no_alt, target_no_alt)
        self.set_heading(math.degrees(bearing))
        self.set_course(math.degrees(bearing))
        speed_km_per_second = self.gps.speed / 1000  # meters/second/1000 = km/second
        speed = speed_km_per_second / self.logic.rate
        new_location = inverse_haversine(current_no_alt, speed, bearing)

        # Change position of the Agent
        self.gps.lat, self.gps.lon = new_location

    def is_task_supported(self, task: dict) -> bool:
        """Check if the task is supported by the agent"""
        name: str = task["name"]
        task_supported: bool = False
        for ava_task in self.logic.tasks_available:
            if name == ava_task["name"]:
                task_supported = True
                break
        return task_supported

    def check_task(self) -> None:
        """Check if there is a task to perform and send feedback if it's finished"""
        if self.logic.task_running and not self.logic.task_pause_flag:
            self.move_to_target((self.gps.lat, self.gps.lon, self.gps.alt), self.logic.task_target)

            # Was task completed?
            if not self.logic.task_running:
                json_msg = {
                    "agent-uuid": self.logic.uuid,
                    "com-uuid": str(uuid.uuid4()),
                    "status": "finished",
                    "task-uuid": self.logic.task_running_uuid
                }
                self.logic.task_running_uuid = ""

                str_msg = json.dumps(json_msg)
                self.publish(f'{self.mqtt_client.base_topic}/exec/feedback', str_msg)

    def bearing(self, current: list, target: list) -> float:
        """Calculate bearing between two points"""
        startLat = math.radians(current[0])
        startLng = math.radians(current[1])
        destLat = math.radians(target[0])
        destLng = math.radians(target[1])
        y = math.sin(destLng - startLng) * math.cos(destLat)
        x = math.cos(startLat) * math.sin(destLat) - math.sin(startLat) * math.cos(destLat) * math.cos(destLng - startLng)
        brng = math.atan2(y, x)
        return brng
