#Pub
#!/usr/bin/env python3
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge
import cv2 # OpenCV library
def publish_message():
   pub = rospy.Publisher('video_frames', Image, queue_size=10)
   # Tells rospy the name of the node.
   # Anonymous = True makes sure the node has a unique name. Random
   # numbers are added to the end of the name.
   rospy.init_node('video_pub_py', anonymous=True)
   # Go through the loop 10 times per second
   rate = rospy.Rate(10) # 10hz
   # Create a VideoCapture object
   # The argument '0' gets the default webcam.
   cap = cv2.VideoCapture(0)
   # Used to convert between ROS and OpenCV images
   br = CvBridge()
   # While ROS is still running.
   while not rospy.is_shutdown():
   ret, frame = cap.read()
   if ret == True:
   # Print debugging information to the terminal
   rospy.loginfo('publishing video frame')
   # Publish the image.
   # The 'cv2_to_imgmsg' method converts an OpenCV
   # image to a ROS image message
   image = cv2.putText(frame, 'A', (150,150), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0, 0), 2, cv2.LINE_AA)
   pub.publish(br.cv2_to_imgmsg(image))
   # Sleep just enough to maintain the desired rate
   rate.sleep()

if __name__ == '__main__':
   try:
   publish_message()
   except rospy.ROSInterruptException:
   pass

#Sub
#!/usr/bin/env python3
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and
OpenCV Images
import cv2 # OpenCV library
def callback(data):
   # Used to convert between ROS and OpenCV images
   br = CvBridge()
   # Output debugging information to the terminal
   rospy.loginfo("receiving video frame")

   # Convert ROS Image message to OpenCV image
   current_frame = br.imgmsg_to_cv2(data)

   # Display image
   cv2.imshow("camera", current_frame)

   cv2.waitKey(1)

def receive_message():
   # Tells rospy the name of the node.
   # Anonymous = True makes sure the node has a unique name. Random
   # numbers are added to the end of the name.
   rospy.init_node('video_sub_py', anonymous=True)

   # Node is subscribing to the video_frames topic
   rospy.Subscriber('video_frames', Image, callback)
   # spin() simply keeps python from exiting until this node is stopped
   rospy.spin()
   # Close down the video stream when done
   cv2.destroyAllWindows()

if __name__ == '__main__':
   receive_message()
   
#TurtleSim
#roscore
#rosrun turtlesim turtlesim_node
#rosrun turtlesim turtle_teleop_key

