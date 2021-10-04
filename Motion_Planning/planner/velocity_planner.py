#!/usr/bin/env python3


import numpy as np
from math import sin, cos, pi, sqrt

class VelocityPlanner:
    def __init__(self, time_gap, a_max, slow_speed, stop_line_buffer):
        self._time_gap         = time_gap
        self._a_max            = a_max
        self._slow_speed       = slow_speed
        self._stop_line_buffer = stop_line_buffer
        self._prev_trajectory  = [[0.0, 0.0, 0.0]]

    def get_open_loop_speed(self, timestep):
        if len(self._prev_trajectory) == 1:
            return self._prev_trajectory[0][2] 

        if timestep < 1e-4:
            return self._prev_trajectory[0][2]

        for i in range(len(self._prev_trajectory)-1):
            distance_step = np.linalg.norm(np.subtract(self._prev_trajectory[i+1][0:2], 
                                                       self._prev_trajectory[i][0:2]))
            velocity = self._prev_trajectory[i][2]
            time_delta = distance_step / velocity

            if time_delta > timestep:
                v1 = self._prev_trajectory[i][2]
                v2 = self._prev_trajectory[i+1][2]
                v_delta = v2 - v1
                interpolation_ratio = timestep / time_delta
                return v1 + interpolation_ratio * v_delta

            else:
                timestep -= time_delta


        return self._prev_trajectory[-1][2]


    def compute_velocity_profile(self, path, desired_speed, ego_state, 
                                 closed_loop_speed, decelerate_to_stop, 
                                 lead_car_state, follow_lead_vehicle):
        """Computes the velocity profile for the local planner path.
        
        args:
            path: Path (global frame) that the vehicle will follow.
                Format: [x_points, y_points, t_points]
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith point's y value:
                        paths[1][i]
                It is assumed that the stop line is at the end of the path.
            desired_speed: speed which the vehicle should reach (m/s)
            ego_state: ego state vector for the vehicle, in the global frame.
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
            decelerate_to_stop: Flag where if true, should decelerate to stop
            lead_car_state: the lead vehicle current state.
                Format: [lead_car_x, lead_car_y, lead_car_speed]
                    lead_car_x and lead_car_y   : position (m)
                    lead_car_speed              : lead car speed (m/s)
            follow_lead_vehicle: If true, the ego car should perform lead
                vehicle handling, as the lead vehicle is close enough to
                influence the speed profile of the local path.
        internal parameters of interest:
            self._slow_speed: coasting speed (m/s) of the vehicle before it 
                comes to a stop
            self._stop_line_buffer: buffer distance to stop line (m) for vehicle
                to stop at
            self._a_max: maximum acceleration/deceleration of the vehicle (m/s^2)
            self._time_gap: Amount of time taken to reach the lead vehicle from
                the current position
        returns:
            profile: Updated profile which contains the local path as well as
                the speed to be tracked by the controller (global frame).
                Length and speed in m and m/s.
                Format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...,
                         [xm, ym, vm]]

        """
        profile = []

        start_speed = ego_state[3]
        if decelerate_to_stop:
            profile = self.decelerate_profile(path, start_speed)


        elif follow_lead_vehicle:
            profile = self.follow_profile(path, start_speed, desired_speed, 
                                          lead_car_state)


        else:
            profile = self.nominal_profile(path, start_speed, desired_speed)

        if len(profile) > 1:
            interpolated_state = [(profile[1][0] - profile[0][0]) * 0.1 + profile[0][0], 
                                  (profile[1][1] - profile[0][1]) * 0.1 + profile[0][1], 
                                  (profile[1][2] - profile[0][2]) * 0.1 + profile[0][2]]
            del profile[0]
            profile.insert(0, interpolated_state)

        self._prev_trajectory = profile

        return profile

    def decelerate_profile(self, path, start_speed): 
        """Computes the velocity profile for the local path to decelerate to a
        stop.
        
        args:
            path: Path (global frame) that the vehicle will follow.
                Format: [x_points, y_points, t_points]
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith point's y value:
                        paths[1][i]
                It is assumed that the stop line is at the end of the path.
            start_speed: speed which the vehicle starts with (m/s)
        internal parameters of interest:
            self._slow_speed: coasting speed (m/s) of the vehicle before it 
                comes to a stop
            self._stop_line_buffer: buffer distance to stop line (m) for vehicle
                to stop at
            self._a_max: maximum acceleration/deceleration of the vehicle (m/s^2)
        returns:
            profile: deceleration profile which contains the local path as well
                as the speed to be tracked by the controller (global frame).
                Length and speed in m and m/s.
                Format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...,
                         [xm, ym, vm]]
        """
        profile          = []
        slow_speed       = self._slow_speed
        stop_line_buffer = self._stop_line_buffer


        decel_distance = calc_distance(start_speed, slow_speed, -self._a_max)
        brake_distance = calc_distance(slow_speed, 0, -self._a_max)

        path_length = 0.0
        for i in range(len(path[0])-1):
            path_length += np.linalg.norm([path[0][i+1] - path[0][i], 
                                           path[1][i+1] - path[1][i]])

        stop_index = len(path[0]) - 1
        temp_dist = 0.0
        # Compute the index at which we should stop.
        while (stop_index > 0) and (temp_dist < stop_line_buffer):
            temp_dist += np.linalg.norm([path[0][stop_index] - path[0][stop_index-1], 
                                         path[1][stop_index] - path[1][stop_index-1]])
            stop_index -= 1

        if brake_distance + decel_distance + stop_line_buffer > path_length:
            speeds = []
            vf = 0.0
            for i in reversed(range(stop_index, len(path[0]))):
                speeds.insert(0, 0.0)
            for i in reversed(range(stop_index)):
                dist = np.linalg.norm([path[0][i+1] - path[0][i], 
                                       path[1][i+1] - path[1][i]])
                vi = calc_final_speed(vf, self._a_max, dist) 

                if vi > start_speed:
                    vi = start_speed

                speeds.insert(0, vi)
                vf = vi

            for i in range(len(speeds)):
                profile.append([path[0][i], path[1][i], speeds[i]])

        else:
            brake_index = stop_index 
            temp_dist = 0.0
            while (brake_index > 0) and (temp_dist < brake_distance):
                temp_dist += np.linalg.norm([path[0][brake_index] - path[0][brake_index-1], 
                                             path[1][brake_index] - path[1][brake_index-1]])
                brake_index -= 1

     
            decel_index = 0
            temp_dist = 0.0
            while (decel_index < brake_index) and (temp_dist < decel_distance):
                temp_dist += np.linalg.norm([path[0][decel_index+1] - path[0][decel_index], 
                                             path[1][decel_index+1] - path[1][decel_index]])
                decel_index += 1


            vi = start_speed
            for i in range(decel_index): 
                dist = np.linalg.norm([path[0][i+1] - path[0][i], 
                                       path[1][i+1] - path[1][i]])
                vf = calc_final_speed(vi, -self._a_max, dist)
                if vf < slow_speed:
                    vf = slow_speed

                profile.append([path[0][i], path[1][i], vi])
                vi = vf

            for i in range(decel_index, brake_index):
                profile.append([path[0][i], path[1][i], vi])

            for i in range(brake_index, stop_index):
                dist = np.linalg.norm([path[0][i+1] - path[0][i], 
                                       path[1][i+1] - path[1][i]])
                vf = calc_final_speed(vi, -self._a_max, dist)
                profile.append([path[0][i], path[1][i], vi])
                vi = vf

            for i in range(stop_index, len(path[0])):
                profile.append([path[0][i], path[1][i], 0.0])

        return profile

    # Computes a profile for following a lead vehicle..
    def follow_profile(self, path, start_speed, desired_speed, lead_car_state):
        """Computes the velocity profile for following a lead vehicle.
        
        args:
            path: Path (global frame) that the vehicle will follow.
                Format: [x_points, y_points, t_points]
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith point's y value:
                        paths[1][i]
                It is assumed that the stop line is at the end of the path.
            start_speed: speed which the vehicle starts with (m/s)
            desired_speed: speed which the vehicle should reach (m/s)
            lead_car_state: the lead vehicle current state.
                Format: [lead_car_x, lead_car_y, lead_car_speed]
                    lead_car_x and lead_car_y   : position (m)
                    lead_car_speed              : lead car speed (m/s)
        internal parameters of interest:
            self._a_max: maximum acceleration/deceleration of the vehicle (m/s^2)
            self._time_gap: Amount of time taken to reach the lead vehicle from
                the current position
        returns:
            profile: Updated follow vehicle profile which contains the local
                path as well as the speed to be tracked by the controller 
                (global frame).
                Length and speed in m and m/s.
                Format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...,
                         [xm, ym, vm]]

        """
        profile = []
        min_index = len(path[0]) - 1
        min_dist = float('Inf')
        for i in range(len(path)):
            dist = np.linalg.norm([path[0][i] - lead_car_state[0], 
                                   path[1][i] - lead_car_state[1]])
            if dist < min_dist:
                min_dist = dist
                min_index = i

        desired_speed = min(lead_car_state[2], desired_speed)
        ramp_end_index = min_index
        distance = min_dist
        distance_gap = desired_speed * self._time_gap
        while (ramp_end_index > 0) and (distance > distance_gap):
            distance += np.linalg.norm([path[0][ramp_end_index] - path[0][ramp_end_index-1], 
                                        path[1][ramp_end_index] - path[1][ramp_end_index-1]])
            ramp_end_index -= 1


        if desired_speed < start_speed:
            decel_distance = calc_distance(start_speed, desired_speed, -self._a_max)
        else:
            decel_distance = calc_distance(start_speed, desired_speed, self._a_max)

        vi = start_speed
        for i in range(ramp_end_index + 1):
            dist = np.linalg.norm([path[0][i+1] - path[0][i], 
                                   path[1][i+1] - path[1][i]])
            if desired_speed < start_speed:
                vf = calc_final_speed(vi, -self._a_max, dist)
            else:
                vf = calc_final_speed(vi, self._a_max, dist)

            profile.append([path[0][i], path[1][i], vi])
            vi = vf

        for i in range(ramp_end_index + 1, len(path[0])):
            profile.append([path[0][i], path[1][i], desired_speed])

        return profile

    def nominal_profile(self, path, start_speed, desired_speed):
        """Computes the velocity profile for the local planner path in a normal
        speed tracking case.
        
        args:
            path: Path (global frame) that the vehicle will follow.
                Format: [x_points, y_points, t_points]
                        x_points: List of x values (m)
                        y_points: List of y values (m)
                        t_points: List of yaw values (rad)
                    Example of accessing the ith point's y value:
                        paths[1][i]
                It is assumed that the stop line is at the end of the path.
            desired_speed: speed which the vehicle should reach (m/s)
        internal parameters of interest:
            self._a_max: maximum acceleration/deceleration of the vehicle (m/s^2)
        returns:
            profile: Updated nominal speed profile which contains the local path
                as well as the speed to be tracked by the controller (global frame).
                Length and speed in m and m/s.
                Format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...,
                         [xm, ym, vm]]
        """
        profile = []

        if desired_speed < start_speed:
            accel_distance = calc_distance(start_speed, desired_speed, -self._a_max)
        else:
            accel_distance = calc_distance(start_speed, desired_speed, self._a_max)

        ramp_end_index = 0
        distance = 0.0
        while (ramp_end_index < len(path[0])-1) and (distance < accel_distance):
            distance += np.linalg.norm([path[0][ramp_end_index+1] - path[0][ramp_end_index], 
                                        path[1][ramp_end_index+1] - path[1][ramp_end_index]])
            ramp_end_index += 1

        vi = start_speed
        for i in range(ramp_end_index):
            dist = np.linalg.norm([path[0][i+1] - path[0][i], 
                                   path[1][i+1] - path[1][i]])
            if desired_speed < start_speed:
                vf = calc_final_speed(vi, -self._a_max, dist)
                # clamp speed to desired speed
                if vf < desired_speed:
                    vf = desired_speed
            else:
                vf = calc_final_speed(vi, self._a_max, dist)
                # clamp speed to desired speed
                if vf > desired_speed:
                    vf = desired_speed

            profile.append([path[0][i], path[1][i], vi])
            vi = vf

        # If the ramp is over, then for the rest of the profile we should
        # track the desired speed.
        for i in range(ramp_end_index+1, len(path[0])):
            profile.append([path[0][i], path[1][i], desired_speed])

        return profile


def calc_distance(v_i, v_f, a):
    """Computes the distance given an initial and final speed, with a constant
    acceleration.
    
    args:
        v_i: initial speed (m/s)
        v_f: final speed (m/s)
        a: acceleration (m/s^2)
    returns:
        d: the final distance (m)
    """
    pass


    d = (v_f**2 - v_i**2) / (2 * a)
    return d

def calc_final_speed(v_i, a, d):
    """Computes the final speed given an initial speed, distance travelled, 
    and a constant acceleration.
    
    args:
        v_i: initial speed (m/s)
        a: acceleration (m/s^2)
        d: distance to be travelled (m)
    returns:
        v_f: the final speed (m/s)
    """
    pass

    v_f = np.sqrt(v_i**2 + 2*a*d)
    return v_f

