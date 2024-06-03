import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Twist
import math

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_sum = 0.0
        self.preve_rror = 0.0
    
    def update(self, error):
        output = self.kp * error + self.ki * self.error_sum + self.kd * (error - self.preve_rror)

        self.error_sum += error
        self.preve_rror = error

        return output
    
class HumanFollowingNode(Node):
    def __init__(self):
        super().__init__('human_following')
        # self.subscription = self.create_subscription(
        #     Marker,
        #     '/vector_distance',
        #     self.vector_distance_callback,
        #     10)
        
        self.angle_subscriber = self.create_subscription(
            Marker,
            '/robot_to_leg_vector',
            self.vector_arrows_callback,
            10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.distance = 0.0
        self.angle_degrees = 0.0

        self.distance_controller = PIDController(kp=1.0, ki=0.0, kd=0.1)
        self.angle_controller = PIDController(kp=0.05, ki=0.0, kd=0.1)

    # def vector_distance_callback(self, msg):
    #     if msg.text:
    #         try: 
    #             self.distance = float(msg.text[:-1])     
    #             self.get_logger().info("Received distance value: %s" % self.distance)

    #             control_output = self.distance_controller.update(self.distance-0.4)

    #             self.get_logger().info("Distance control output: %s" % control_output)
                
    #             twist_msg = Twist()
    #             twist_msg.linear.x = control_output  # Set linear velocity
    #             twist_msg.angular.z = 0.0  # Set angular velocity
    #             # self.cmd_vel_publisher.publish(twist_msg)

    #         # Publish Twist message
                

    #         except ValueError:
    #             self.get_logger().error("Invalid distance value: %s" % msg.text)
    #     else:
    #         self.get_logger().warn("Empty arrows message received")
        # self.publish_distance_and_angle()
    
    

    def vector_arrows_callback(self, msg):
        if len(msg.points) >= 2:
            try:
                self.second_point = msg.points[1]
                x = round(self.second_point.x, 2)
                y = round(self.second_point.y, 2)
                print("Second point coordinates (x, y):", x, y)
                
                magnitude_a = math.sqrt(x ** 2 + y ** 2)
                magnitude_b = math.sqrt(0 ** 2 + 1 ** 2)

                magnitude_a = round(magnitude_a, 2)
                self.get_logger().info(f"Received distance value: {magnitude_a}"  )
                
                self.distance=magnitude_a-0.5
                control_output_DS = self.distance_controller.update(self.distance)       

                if magnitude_a !=0 :

                    dot_product = x * 0 + y * -1

                    angle_radians = math.acos(dot_product / (magnitude_a * magnitude_b))

                    self.angle_degrees = math.degrees(angle_radians)
                    self.angle_degrees = round(self.angle_degrees, 2)
                    
                    self.get_logger().info(f"Angle between the target point and the robot : {self.angle_degrees}")
                    if x<0:
                        self.angle_degrees*=-1

                    control_output_AG = self.angle_controller.update(self.angle_degrees)

                    self.get_logger().info(f"Distance control output: {control_output_DS}")
                    self.get_logger().info(f'''Angle control output: {control_output_AG}''')


                    twist_msg= Twist()
                    twist_msg.linear.x = control_output_DS  # Set linear velocity
                    twist_msg.angular.z = control_output_AG  # Set angular velocity left is + right is -

                    # Publish Twist message
                    self.cmd_vel_publisher.publish(twist_msg)
                    
            except IndexError:
                self.get_logger().error("Failed to access second point in the message")
                x = 0  
                y = 0  

  

    

def main(args=None):
    rclpy.init(args=args)
    human_following_node = HumanFollowingNode()
    rclpy.spin(human_following_node)
    human_following_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
