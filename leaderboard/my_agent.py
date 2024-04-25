import carla
from leaderboard.autoagents.autonomous_agent import AutonomousAgent
from leaderboard.autoagents.autonomous_agent import Track

def get_entry_point():
    return 'MyAgent'

class MyAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.track = Track.MAP

    def sensors(self):
        sensors = [{
                'type': 'sensor.camera.rgb',
                'id': 'Center',
                'x': 0.7,
                'y': 0.0,
                'z': 1.60,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'width': 300,
                'height': 200,
                'fov': 100
            }, {
                'type': 'sensor.opendrive_map',
                'id': 'OpenDRIVE',
                'reading_frequency': 1
            }, {
                'type': 'sensor.speedometer',
                'id': 'Speed'
            },
        ]
        return sensors

    def run_step(self, input_data, timestamp):
        #control = self._do_something_smart(input_data, timestamp)
        print(input_data)
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 1.0
        control.brake = 0.0
        control.hand_brake = False
        return control

    def destroy(self):
        pass
