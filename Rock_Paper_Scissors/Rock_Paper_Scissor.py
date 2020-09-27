import cv2
import numpy as np
from random import choice
from tensorflow.keras.models import load_model
from preprocess_and_train import preprocess



REV_CLASS_MAP = {
    0: "empty",
    1: "rock",
    2: "paper",
    3: "scissors"
}


def mapper(value):
    return REV_CLASS_MAP[value]

# Function implementing the logic of rock paper scissors game
def find_winner(move_made_by_you,move_made_by_bot):
    win = "Winer Declaration"
    if move_made_by_you == 'scissors':
            if move_made_by_bot == 'scissors':
                win = "__Match_Draw__"
            elif move_made_by_bot == 'paper':
                win = "You"
            elif move_made_by_bot == 'rock':
                win = "Bot Player"
                
                
    if move_made_by_you == 'rock':
            if move_made_by_bot == 'rock':
                win = "__Match_Drawn__"
            elif move_made_by_bot == 'scissors':
                win = "You"
            elif move_made_by_bot == 'paper':
                win = "Bot Player"
                
    if move_made_by_you == 'paper':
            if move_made_by_bot == 'paper':
                win = "_Match_Drawn__"
            elif move_made_by_bot == 'rock':
                win = "You"
            elif move_made_by_bot == 'scissors':
                win = "Bot Player"
                
    if move_made_by_you == 'empty':
                win = "User Did'nt Showed His Hands"
    return win
    

def main():
    
    #Loading the model for detecting the actions performed by user on camera
  
    model = load_model("rock-paper-scissors-model.h5")

    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)

    if not cap.isOpened():
        print("Error opening video")


    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # rectangle for input sub-frame
        cv2.rectangle(frame, (75, 75), (500, 500), (0, 0, 255), 2)

        # extract the region of image within the input sub-frame
        capture_region = frame[75:500, 75:500]
        img = cv2.cvtColor(capture_region, cv2.COLOR_BGR2RGB)
        img = preprocess(img)

        # predict the move made by user on camera using trained model
        pred = model.predict(np.array([img]))
        user_move = mapper(np.argmax(pred[0]))

        winner = None
        computer_move = None
        
        if user_move != 'empty':
            # Selecting random action for the computer or bot player
            computer_move = choice(['rock','paper','scissors'])
            winner = find_winner(user_move, computer_move)
        else:
            computer_move = 'empty'
            winner = 'waiting...'
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Your Move: " + user_move, (110, 50), font, 1.2, (0,0,0), 2)
        cv2.putText(frame, "Press q to exit", (400, 600), font, 2, (242, 186, 157), 4, cv2.LINE_AA)
        cv2.imshow("Rock Paper Scissors", frame)
        
        k = cv2.waitKey(10)
        if k == ord('q'):
            lst = [user_move,computer_move,winner]
            break
    cap.release()
    cv2.destroyAllWindows()
    return lst

result = main()
print("\nYour Move ==========> "+str(result[0]))
print("\nBot's Move ==========> "+str(result[1]))
print("\n-----------********-------------\nThe Winner Is ========>>>>>>> "+str(result[2]))
        
