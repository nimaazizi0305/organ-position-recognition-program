import cv2
import mediapipe as mp


mpDraw=mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
mpPose=mp.solutions.pose
pose=mpPose.Pose()

mpHands = mp.solutions.hands
Hands = mpHands.Hands()
mpDraw_hand = mp.solutions.drawing_utils


while True:
    img,frame=cap.read()

    frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=pose.process(frameRGB)
    results = Hands.process(frameRGB)


    if result.pose_landmarks:
        mpDraw.draw_landmarks(frame,result.pose_landmarks,mpPose.POSE_CONNECTIONS)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame,handlms,mpHands.HAND_CONNECTIONS)


    cv2.imshow("pose Estimation",frame)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()





















