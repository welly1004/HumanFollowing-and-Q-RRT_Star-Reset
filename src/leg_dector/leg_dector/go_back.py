import rclpy
import rclpy.duration
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped,Twist
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Path
import math
import random
from std_msgs.msg import Bool  # 引入 Bool 消息類型
from std_msgs.msg import String
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from nav_msgs.msg import Odometry
from tf2_geometry_msgs import do_transform_point
import tf_transformations
###-*-------------------------------------------------------


import copy
import time


# rs = 42 #randomseed number
# random.seed()


# Function to transform obstacles into rectangles
def convert_obstacles_to_rectangles(grid, rect_size=(2, 2)):
    new_grid = grid.copy()
    rect_h, rect_w = rect_size
    obstacle_indices = np.argwhere((grid == 100) | (grid==-1))
    rectangles = []
    for idx in obstacle_indices:
        x, y = idx
        # Define the rectangle area (clipped to grid boundaries)
        x_start = max(0, x - rect_h // 2)
        x_end = min(grid.shape[0], x + rect_h // 2)
        y_start = max(0, y - rect_w // 2)
        y_end = min(grid.shape[1], y + rect_w // 2)
        # Fill the rectangle with obstacle values
        new_grid[x_start:x_end, y_start:y_end] = 100
        # Append rectangles in the format (x_start, y_start, width, height)
        rectangles.append((x_start,y_start , y_end - y_start, x_end - x_start))
    return new_grid, rectangles

class Q_RRT_STAR:
    def __init__(
        self,
        rand_area=[-4,40],
        map = 'room 2',
        expand_dis=5.0,
        goal_sample_rate=30,
        animation = True,
        ):
        self.start = None
        self.goal = None
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.map = map
        self.node_list = None
        self.animation = animation
        self.depth = 5
        
        
    
    def rrt_planning(self,start,goal,rect_list,max_iter=15000):
       
        #-----------------------------------------#

        self.real_ob_list=rect_list
        # self.obstacle_list, self.real_ob_list = rec_list,real_list#mymap.create_map(self.map)
        self.obstacle_list = self.real_ob_list

        self.max_iter = max_iter
        self.start = rrt_Node(start[0],start[1])
        self.start.index = 0
        self.goal = rrt_Node(goal[0],goal[1])
        self.node_list = [self.start]
        path = None
        s = time.time()
        solution_cost = []
        times = []
        first = 0
        for i in range(self.max_iter):
            # 環境中採點
            rnd = self.sample()
            # 尋找距離採點最近的node
            n_ind = self.get_nearest_list_index(self.node_list,rnd)
            nearst_node = self.node_list[n_ind]
            
            # 由距離最近的 node 往採點方向伸長一個步長，得到新的 node
            theta = math.atan2(rnd[1]-nearst_node.y,rnd[0]-nearst_node.x)
            new_node = self.get_new_node(theta,nearst_node)

            # 碰撞檢測
            no_colllision = self.check_segment_collision(new_node.x,new_node.y,nearst_node.x,nearst_node.y)
            if no_colllision:
                self.node_list.append(new_node)
                # 重新佈線
                self.rewire(new_node)
                # 繪製
                if self.animation:
                    # time.sleep(0.1)
                    self.draw_graph(new_node,path)
                if self.goal.parent is None:
                    if self.is_near_goal(new_node):
                        if self.check_segment_collision(new_node.x,new_node.y,
                                                        self.goal.x,self.goal.y):
                            self.goal.index = len(self.node_list)
                            # print(self.goal.index)
                            self.goal.parent = new_node.index
                            self.node_list.append(self.goal)
                            self.rewire(self.goal)
                            e=time.time()
                            path, path_length = self.get_final_course(self.goal)
                            solution_cost.append(path_length)
                            times.append(e-s)
                            first = i
                else :
                    path, path_length = self.get_final_course(self.goal)
                    e=time.time()
                    solution_cost.append(path_length)
                    times.append(e-s)
                    # print(path_length)
                    # print(e-s)
            if path is not None:
                plt.plot([x for (x,y) in path],[y for (x,y) in path],'-r')
                plt.pause(0.1)
                return path, [1,times[0],solution_cost[0],solution_cost[-1],first]
                

        # plt.figure(figsize=(8,8))
        # self.draw_graph(path=path)
        # plt.show()
        

        # plt.figure(figsize=(8,8))
        # plt.plot(times, solution_cost,'-')
        if len(solution_cost) == 0:
            # plt.title(f"Q-RRT* {' '*23} RandomSeed : {rs:<10} \n First iter : {max_iter:<4}           Time :           sec \n Cost : Planning fail",loc='left',fontsize=20)
            return path, [0,0,0,0,0]
        else:
            # plt.title(f"Q-RRT* {' '*23} RandomSeed : {rs:<10} \n First iter : {first:<4}            Time : {times[0]:.3f} sec \n Optimal Cost : {solution_cost[-1]:.3f}",loc='left',fontsize=20)
            return path, [1,times[0],solution_cost[0],solution_cost[-1],first]
        # plt.show()
        # plt.figure(figsize=(8,8))
        # self.draw_graph(path=path)
        # plt.show()

        return self.node_list
    
    def is_near_goal(self,node):
        d = self.line_cost(node,self.goal)
        if d < self.expand_dis:
            return True
        return False            
                                    
    def sample(self):
        # 選擇 random node
        #一定機率選擇 goal 為 random node
        if random.randint(0,100) > self.goal_sample_rate:
            rnd = [random.uniform(self.min_rand,self.max_rand),random.uniform(self.min_rand,self.max_rand)]
        else:
            rnd = [self.goal.x,self.goal.y]
            # print(f"rnd: ({self.goal.x}, {self.goal.y})")
        return rnd

    @staticmethod
    def get_nearest_list_index(nodes,rnd):
        # 計算各node與採樣點之間的距離
        d_list = [(node.x-rnd[0])**2+(node.y-rnd[1])**2
                    for node in nodes]
        min_index = d_list.index(min(d_list))
        return min_index
    
    def get_new_node(self,theta,nearest_node):
        new_node = copy.deepcopy(nearest_node)
        new_node.cost = 0.0
        new_node.x +=self.expand_dis*math.cos(theta)# *random.random()
        new_node.y +=self.expand_dis*math.sin(theta)# *random.random()
        new_node.index = len(self.node_list)
        nearest_index = self.get_nearest_list_index(
            self.node_list,[new_node.x,new_node.y])
        new_node.parent = nearest_index
        
        
        return new_node
    
    def find_parent_list(self,node,depth):
        parent_list = []
        for i in range(depth):
            if node.parent is not None:
                idx = node.parent
                node = self.node_list[idx]
                parent_list.append(node)
            else:
                break
        return parent_list
    
    def rewire(self,newnode):
        
        # 重新佈線
        re_list = []

        #計算newnode路徑長
        newnode.cost = self.path_cost(newnode.index)

        #尋找距離newnode 2*expan_dis 內的結點 -> re_list[結點index ,結點cost + 結點到newnode的距離 ]
        for node in self.node_list:
            l_c = self.line_cost(newnode,node)
            if (l_c < 2*self.expand_dis) and (l_c > 0):
                idx =node.index
                node.cost = self.path_cost(node.index) 
                re_list.append([idx,node.cost+l_c])
                
        #sort -> 距離短到長
        re_list.sort(reverse=False,key=self.takeSecond)

        #查看 "newnode" 透過"附近結點"是否有更短路徑長
        for idx in re_list:
            min_index = idx[0]
            no_colllision = self.check_segment_collision(newnode.x,newnode.y,self.node_list[min_index].x,self.node_list[min_index].y)
            if idx[1] < newnode.cost:
                if no_colllision:
                    newnode.parent = min_index
                    newnode.cost = self.path_cost(newnode.index)
                    break
        
        # 查看"附近結點"透過 "newnode" 是否有更短的路徑長
        for idx in re_list:
            l_c = self.line_cost(newnode,self.node_list[idx[0]])
            if newnode.cost + l_c < self.node_list[idx[0]].cost:
                no_colllision = self.check_segment_collision(newnode.x,newnode.y,self.node_list[idx[0]].x,self.node_list[idx[0]].y)
                if no_colllision:
                    self.node_list[idx[0]].parent = newnode.index
                    self.node_list[idx[0]].cost = self.path_cost(self.node_list[idx[0]].index)

        #尋回父結點的路徑 -> depth = 尋找深度
        parent_list = self.find_parent_list(newnode,self.depth)
        parent_list.reverse()

        #依照三角不等式的原理，直接找尋路徑上的父結點作為newnode的父結點路徑長會更短
        for parent_node in parent_list:
            no_colllision = self.check_segment_collision(newnode.x,newnode.y,parent_node.x,parent_node.y)
            if no_colllision:
                newnode.parent = parent_node.index
                break
        
        for idx in re_list:
            if idx[0] != newnode.parent:
                l_c = self.line_cost(self.node_list[newnode.parent],self.node_list[idx[0]])
                if self.node_list[newnode.parent].cost + l_c < self.node_list[idx[0]].cost:
                    no_colllision = self.check_segment_collision(self.node_list[newnode.parent].x,self.node_list[newnode.parent].y,self.node_list[idx[0]].x,self.node_list[idx[0]].y)
                    if no_colllision:
                        self.node_list[idx[0]].parent = self.node_list[newnode.parent].index
                        self.node_list[idx[0]].cost = self.path_cost(self.node_list[idx[0]].index)
        

    def takeSecond(self,elem):
        return elem[1]

    def path_cost(self,index):
        # 結點路徑長
        total_cost = 0.0
        while(1):
            if self.node_list[index].parent is not None:
                node = self.node_list[index]
                total_cost = total_cost + self.line_cost(node,self.node_list[node.parent])
                index = node.parent
            else :
                break
        return total_cost
                           
    def check_segment_collision(self,x1,y1,x2,y2):
        # 碰撞檢測矩形
        for (x,y,w,h) in self.obstacle_list:
            # if x1>x and x1<x+w :
                # if y1>y and y1<y+h:
                    # return False 
            if self.judge2(x1,y1,x2,y2,x,y,x+w,y):
                return False
            if self.judge2(x1,y1,x2,y2,x,y,x,y+h):
                return False
            if self.judge2(x1,y1,x2,y2,x,y+h,x+w,y+h):
                return False
            if self.judge2(x1,y1,x2,y2,x+w,y,x+w,y+h):
                return False

        return True

    def judge(self,a,b,c,d):
        ab=b-a
        ac=c-a
        ad=d-a
        bd=d-b
        ca=a-c
        cb=b-c
        cd=d-c
        if abs(ab[0])+abs(ab[1])==0:
            return True
        abc = self.cross2(ab,ac)
        abd = self.cross2(ab,ad)
        cda = self.cross2(cd,ca)
        cdb = self.cross2(cd,cb)   
        if abc*abd<0 and cda*cdb<0:
          return True
        elif abc==0:
          if abs(ac[0])+abs(ac[1])+abs(cb[0])+abs(cb[1])==abs(ab[0])+abs(ab[1]):
            return True
        elif abd==0:
          if abs(ad[0])+abs(ad[1])+abs(bd[0])+abs(bd[1])==abs(ab[0])+abs(ab[1]):
            return True
        # 未相交
        return False

    def cross2(self,v1,v2)->np.array:
      return np.cross(v1,v2)

    def judge2(self,x1,y1,x2,y2,ox1,oy1,ox2,oy2):
        det = (oy2-oy1)*(x2-x1) - (ox2-ox1)*(y2-y1)
        if det == 0:
            return self.judge(np.array([x1,y1]),np.array([x2,y2]),np.array([ox1,oy1]),np.array([ox2,oy2]))
        mA = ((ox2-ox1)*(y1-oy1) - (oy2-oy1)*(x1-ox1)) / det
        mB = ((x2-x1)*(y1-oy1) - (y2-y1)*(x1-ox1)) / det
        if mA>=0 and mA<=1 and mB>=0 and mB<=1:
            return True
        return False

    @staticmethod
    def distace_point_to_segment(v,w,p):
        vw = w-v
        vp = p-v
        wp = p-w
        disvw = math.sqrt(vw.dot(vw))
        if disvw == 0:
          return 1000
        disvp = math.sqrt(vp.dot(vp))
        diswp = math.sqrt(wp.dot(wp))
        r = (vw.dot(vp))/(disvw**2)
        # print(f"AB:{AB} ,AP:{AP} ,disAP:{disAP} ,disAB:{disAB}}}")
        # print((AB[0]*AP[0]+AB[1]*AP[1])/(disAP*disAB))
        theta = math.acos(vw.dot(vp)/(disvp*disvw))
        if r <= 0 :
              distance = disvp
        elif r < 1 :
              distance = disvp*math.sin(theta)
        else :
              distance = diswp
        return distance
        # if np.array_equal(v,w):
            # return (p-v).dot(w-v)
        # 
        # l2 = (w-v).dot(w-v)
        # t = max(0,min(1,(p-v).dot(w-v)/l2))
        # projection = v+t*(w-v)
        # return (p-projection).dot(p-projection)

    def draw_graph(self,rnd=None,path=None):
        plt.clf()
        # plt.axis('square')
        # draw node
        if rnd is not None:
            plt.plot(rnd.x,rnd.y,'^k')
            
        #draw obstical
        for (x,y,w,h) in self.obstacle_list:
            rectangle = plt.Rectangle((x,y),w,h,color='aqua')
            plt.gca().add_patch(rectangle)

        for (x,y,w,h) in self.real_ob_list:
            rectangle = plt.Rectangle((x,y),w,h,color='black')
            plt.gca().add_patch(rectangle)
        # draw path
        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x,self.node_list[node.parent].x],
                            [node.y,self.node_list[node.parent].y],
                             '-g')
                    plt.plot([node.x,self.node_list[node.parent].x],
                            [node.y,self.node_list[node.parent].y],
                            '.g')
                    # plt.annotate(node.parent,(node.x,node.y),color='red')
                    # plt.annotate(f"{node.index}",(node.x,node.y),color='blue')

        # draw start, goal
        plt.plot(self.start.x,self.start.y,'ob')
        plt.annotate(' Start',(self.start.x,self.start.y),color='blue')
        plt.plot(self.goal.x,self.goal.y,'or')
        plt.annotate(' Goal',(self.goal.x,self.goal.y),color='red')

        # draw final path
        if path is not None:
            plt.plot([x for (x,y) in path],[y for (x,y) in path],'-r')

        # graph setting
        # plt.axis([-2,20,-2,20])
        plt.grid()
        ax=plt.gca()
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        plt.pause(0.01)
    
    # def is_near_goal(self,node):
        # d = self.line_cost(node,self.goal)
        # if d < self.expand_dis:
            # return True
        # return False

    @staticmethod
    def line_cost(node1,node2):
        return math.sqrt((node1.x-node2.x)**2+(node1.y-node2.y)**2)

    def get_final_course(self,node):
        # 路徑回溯
        path = [[node.x,node.y]]
        path_length = 0
        last_index = node.parent
        while node.parent is not None:
            nodep = self.node_list[node.parent]
            path.append([nodep.x,nodep.y])
            path_length += self.line_cost(node,nodep)
            node = self.node_list[node.parent]
        return path,path_length

    
class rrt_Node:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None
        self.index = None


###-*-------------------------------------------------------
class QRRTStar(Node):
    def __init__(self):
        super().__init__('qrrt_star_node')
        
        self.path_pub = self.create_publisher(Path, 'qrrt_star_path', 10)
        
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data)

        self.subscription = self.create_subscription(
            PoseStamped,
            '/start_position',
            self.start_position_callback,
            10) 
        
        # 訂閱 go_back_callback
        self.go_back_sub = self.create_subscription(
            Bool,
            'go_back',
            self.go_back_callback,
            qos_profile=rclpy.qos.QoSProfile(depth=10))
        
        
        
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.current_goal_index = 0


        self.current_position = None
        self.current_orientation = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.END = None
        self.START = None

        # 定时器用于周期性调用回调函数
        # self.timer = self.create_timer(1.0, self.timer_callback)
        

        self.map_data = None
        self.start = None
        self.goal = None
        self.map_bool=False
        self.path=None
        self.map_origin=None
        # 初始化起點和終點
        self.start_pose = None
        self.end_pose = None

        # 記錄當前的里程計數據
        self.current_position = None

        self.go_back_bool=False
    

    def start_position_callback(self, msg:PoseStamped):
        # print("goal : ", msg)
       
        self.goal=msg.pose
        
    def odom_callback(self, msg:Odometry):
        self.current_position=[msg.pose.pose.position.x,msg.pose.pose.position.y]
        self.start=msg.pose.pose

        if self.map_origin==None or self.map_bool==False:
            return
        if len(self.path)<=1:
            print("no extra node can go")
            return
        if self.current_goal_index>=len(self.path):
            self.get_logger().warn('已到達路徑終點')
            rclpy.shutdown()
        self.navigate_to_goal(msg.pose.pose)

    def navigate_to_goal(self,current_robot_pose):
        rotation_matrix=tf_transformations.quaternion_matrix( [
        
        current_robot_pose.orientation.x,
        current_robot_pose.orientation.y,
        current_robot_pose.orientation.z,
        current_robot_pose.orientation.w
        ])[:2, :2]
        
        xy1=np.dot(  rotation_matrix,   [0,-1]  )
        xy1=np.dot(  [[0,-1],[1,0]],   xy1  )
        xy1=np.add(xy1,self.current_position)

        
        sg=np.subtract(self.path[self.current_goal_index],self.current_position)
        sh=np.subtract(xy1,self.current_position)


        #compute angel between two vector
        # 計算點積 
        dot_product = np.dot(sg, sh)
        # 計算範數
        sg_dist = np.linalg.norm(sg)
        norm_v2 = np.linalg.norm(sh)
        # 計算夾角的餘弦值
        cos_theta = dot_product / (sg_dist * norm_v2)
        # 計算夾角（以弧度為單位）
        error_angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 使用 clip 來避免數值誤差

        linear_speed = sg_dist*0.7
        angular_speed = 0.5 * error_angle_rad
        if error_angle_rad>=0.1:
            linear_speed=0.0

        # 如果距離足夠近，則進入下一個路徑點
        if sg_dist < 0.1:
            self.current_goal_index += 1
            self.get_logger().info(f'到達路徑點 {self.current_goal_index}, 移動到下一個目標')

        # 發佈速度指令
        self.send_velocity_command(linear_speed, angular_speed)

    
    
#------------------
    # def odom_callback(self, msg):
    #     # 獲取當前的位置信息
    #     self.current_odom = msg.pose.pose
    #     self.start=msg.pose.pose
        

    #     if self.map_origin==None:
    #         return
        
            
    #     # self.current_position=np.add(point_origin,[self.current_odom.position.y*10,self.current_odom.position.x*10])
        

    #     # print("Odometry received: ", self.current_odom)
        
        
    #     self.current_position = [msg.pose.pose.position.x*10, msg.pose.pose.position.y*10]
    #     print(f"Current Position: {self.current_position}")
    #     # 提取四元數並轉換為歐拉角
    #     orientation_q = msg.pose.pose.orientation
    #     orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    #     (roll, pitch, yaw) = tf_transformations.euler_from_quaternion(orientation_list)

    #     self.current_orientation = yaw  # 機器人的朝向（yaw）

    #     # 開始導航

    #     self.navigate_to_goal()
    
    # def navigate_to_goal(self):
    #     if self.map_bool==False:
    #         return
        
        
        
    #     print("navigate_to_goal OK")
        
    #     self.current_goal_index = 0
    #     self.get_logger().info(f'生成的路徑: {self.path}')
    #     if self.current_goal_index >= len(self.path):
    #         self.get_logger().info('已到達路徑終點')
    #         return
        
    #     # 當前目標點
    #     goal = self.path[self.current_goal_index]
        
    #     goal_x, goal_y = goal[0], goal[1]
        
    #     # 計算與目標的距離
    #     # if self.current_position:
    #     dx = goal_x - self.current_position[0]
    #     dy = goal_y - self.current_position[1]

    #     # 計算距離和目標方向（在世界座標系下）
    #     distance_to_goal = math.sqrt(dx**2 + dy**2)
    #     angle_to_goal = math.atan2(dy, dx)

    #     # 計算機器人需要轉向的角度（在機器人座標系下）
    #     angle_diff = (math.pi/2)-((math.pi/2)-angle_to_goal) - self.current_orientation
    #     angle_diff = abs(angle_diff)
    #     if dx<0:
    #         angle_diff*=-1
    #     # 控制參數
    #     linear_speed = 0.01 * distance_to_goal
    #     angular_speed = 0.1 * angle_diff

    #     # 如果距離足夠近，則進入下一個路徑點
    #     if distance_to_goal < 5:
    #         self.current_goal_index += 1
    #         self.get_logger().info(f'到達路徑點 {self.current_goal_index}, 移動到下一個目標')

    #     # 發佈速度指令
    #     self.send_velocity_command(linear_speed, angular_speed)
#-----------------------


    def send_velocity_command(self, linear_speed, angular_speed):
        # 建立並發佈速度指令
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_speed
        cmd_vel_msg.angular.z = angular_speed
        self.cmd_vel_publisher.publish(cmd_vel_msg)

    def map_callback(self, msg:OccupancyGrid):
        self.map_origin=msg.info.origin
        if self.go_back_bool==False or self.map_bool:
            return
        self.rrt = Q_RRT_STAR(rand_area=[0,max(msg.info.height,msg.info.width)])
        # print("map ok")
        
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_data=self.map_data[::,::]
        self.map_info = msg.info


       
        


        # Convert obstacles to rectangles
        rect_grid,rectangles = convert_obstacles_to_rectangles(self.map_data)
        # print(rectangles)
        # # Visualize the original and modified grids
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # # Plot original map
        # ax1.imshow(self.map_data, cmap='gray_r', origin='lower')
        # ax1.set_title('Original Map')
        # ax1.set_xticks([]), ax1.set_yticks([])

        # # Plot map with rectangular obstacles
        # ax2.imshow(rect_grid, cmap='gray_r', origin='lower')
        # ax2.set_title('Rectangular Obstacles')
        # ax2.set_xticks([]), ax2.set_yticks([])

        # plt.show()
        
        # output_path
        if (self.start!=None and self.goal!=None) :
            resolution=1/msg.info.resolution
            self.map_bool=True
            self.get_logger().info("Start To Run Quick-RRT*")
            point_origin=np.array([msg.info.origin.position.y*-resolution,msg.info.origin.position.x*-resolution])
            
            start=np.add(point_origin,[self.start.position.y*resolution,self.start.position.x*resolution])
            goal=np.add(point_origin,[self.goal.position.y*resolution,self.goal.position.x*resolution])
            
            # start = [24.7,17.0]
            # goal = [15.0,19.0]
            path, result= self.rrt.rrt_planning(start=start,goal=goal,rect_list=rectangles,max_iter=1500)
            
            self.path=path[1::-1]
            self.path=np.subtract(self.path,point_origin)
            self.path=np.flip(self.path,1)
            self.path=np.divide(self.path,10.0)
            
            # print(f"Processed Path (after adjusting): {self.path}")
            
            # plt.show()

    def go_back_callback(self, msg):
        self.go_back_bool=msg.data
        if msg.data:
            self.get_logger().info("Received go_back_callback signal: True")
        
            # self.get_logger().info("Initialized Q_RRT_STAR")

        
def main(args=None):
    rclpy.init(args=args)
    node = QRRTStar()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
