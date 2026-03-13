"""
Simple Gesture-Controlled Flappy Bird

Clean implementation with straightforward gesture logic:
- Hold PEACE sign → Bird jumps continuously while held
- Release or show FIST → Bird stops jumping
- Makes it easy to control the bird's flight height
"""

import pygame
from pygame.locals import *
import random
import torch
import cv2
import numpy as np
import threading
from gesture_model import GestureRecognitionCNN

pygame.init()

clock = pygame.time.Clock()
fps = 40  # Reduced from 60 for better gesture detection sync

screen_width = 864
screen_height = 936

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Gesture Flappy Bird')

font = pygame.font.SysFont('Bauhaus 93', 60)
small_font = pygame.font.SysFont('Arial', 18)
white = (255, 255, 255)
green = (0, 255, 0)

# Game variables
ground_scroll = 0
scroll_speed = 3  # Reduced from 4 for slower gameplay
flying = False
game_over = False
pipe_gap = 220  # Much larger gap for easier gesture control
pipe_frequency = 2500  # More time between pipes for easier gameplay
last_pipe = pygame.time.get_ticks() - pipe_frequency
score = 0
pass_pipe = False

# Gesture control - SIMPLE STATE
gesture_can_jump = False  # Can we jump right now?
current_gesture = "fist"
gesture_confidence = 0.0

# Load images
bg = pygame.image.load('img/bg.png')
ground_img = pygame.image.load('img/ground.png')
button_img = pygame.image.load('img/restart.png')


def draw_text(text, font, text_col, x, y):
	img = font.render(text, True, text_col)
	screen.blit(img, (x, y))


def draw_gesture_status():
	"""Show gesture status at bottom of screen"""
	status_text = f"Gesture: {current_gesture.upper()} ({gesture_confidence:.0%})"
	# Show in green if peace is active (either can jump or currently showing peace with high conf)
	is_active = gesture_can_jump or (current_gesture == "peace" and gesture_confidence > 0.80)
	draw_text(status_text, small_font, green if is_active else white, 10, screen_height - 50)
	
	# Show if ready to jump
	if is_active:
		draw_text("Jumping!", small_font, green, 10, screen_height - 25)


def reset_game():
	pipe_group.empty()
	flappy.rect.x = 100
	flappy.rect.y = int(screen_height / 2)
	score = 0
	return score


class Bird(pygame.sprite.Sprite):
	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		self.images = []
		self.index = 0
		self.counter = 0
		for num in range(1, 4):
			img = pygame.image.load(f"img/bird{num}.png")
			self.images.append(img)
		self.image = self.images[self.index]
		self.rect = self.image.get_rect()
		self.rect.center = [x, y]
		self.vel = 0
		self.clicked = False

	def update(self):
		global gesture_can_jump

		if flying == True:
			# Gravity
			self.vel += 0.5
			if self.vel > 8:
				self.vel = 8
			if self.rect.bottom < 768:
				self.rect.y += int(self.vel)

		keys = pygame.key.get_pressed()

		if game_over == False:
			# Jump with mouse, keyboard, OR gesture
			# For gesture: allow continuous jumps while peace is held
			if gesture_can_jump:
				# Jump every 15 frames (~0.375 seconds at 40 FPS) for controllability
				if not hasattr(self, 'gesture_cooldown'):
					self.gesture_cooldown = 0
				
				self.gesture_cooldown -= 1
				if self.gesture_cooldown <= 0:
					self.vel = -10
					self.gesture_cooldown = 15  # Reset cooldown
				self.clicked = True  # Prevent mouse/key interference
			elif (pygame.mouse.get_pressed()[0] == 1 or keys[pygame.K_UP]) and self.clicked == False:
				self.clicked = True
				self.vel = -10
			
			if pygame.mouse.get_pressed()[0] == 0 and not keys[pygame.K_UP] and not gesture_can_jump:
				self.clicked = False

			# Animation
			flap_cooldown = 5
			self.counter += 1
			if self.counter > flap_cooldown:
				self.counter = 0
				self.index += 1
				if self.index >= len(self.images):
					self.index = 0
				self.image = self.images[self.index]

			# Rotation
			self.image = pygame.transform.rotate(self.images[self.index], self.vel * -2)
		else:
			self.image = pygame.transform.rotate(self.images[self.index], -90)


class Pipe(pygame.sprite.Sprite):
	def __init__(self, x, y, position):
		pygame.sprite.Sprite.__init__(self)
		self.image = pygame.image.load("img/pipe.png")
		self.rect = self.image.get_rect()
		if position == 1:
			self.image = pygame.transform.flip(self.image, False, True)
			self.rect.bottomleft = [x, y - int(pipe_gap / 2)]
		elif position == -1:
			self.rect.topleft = [x, y + int(pipe_gap / 2)]

	def update(self):
		self.rect.x -= scroll_speed
		if self.rect.right < 0:
			self.kill()


class Button():
	def __init__(self, x, y, image):
		self.image = image
		self.rect = self.image.get_rect()
		self.rect.topleft = (x, y)

	def draw(self):
		action = False
		pos = pygame.mouse.get_pos()
		if self.rect.collidepoint(pos):
			if pygame.mouse.get_pressed()[0] == 1:
				action = True
		screen.blit(self.image, (self.rect.x, self.rect.y))
		return action


class SimpleGestureDetector:
	"""Simplified gesture detector with auto-reset"""
	
	def __init__(self, model_path="gesture_model.pth"):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# Load model
		self.model = GestureRecognitionCNN(num_classes=2)
		checkpoint = torch.load(model_path, map_location=self.device)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.model.to(self.device)
		self.model.eval()
		
		self.gesture_names = {0: "fist", 1: "peace"}
		self.running = False
		
		# Webcam
		self.cap = cv2.VideoCapture(0)
		if not self.cap.isOpened():
			raise RuntimeError("Cannot open webcam")
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	
	def preprocess_frame(self, frame):
		"""Convert frame to model input"""
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		h, w = gray.shape
		size = min(h, w)
		start_h = (h - size) // 2
		start_w = (w - size) // 2
		roi = gray[start_h:start_h+size, start_w:start_w+size]
		resized = cv2.resize(roi, (64, 64))
		equalized = cv2.equalizeHist(resized)
		tensor = torch.from_numpy(equalized).float() / 255.0
		tensor = tensor.unsqueeze(0).unsqueeze(0)
		return tensor.to(self.device)
	
	def detect_loop(self):
		"""Main detection loop"""
		global gesture_can_jump, current_gesture, gesture_confidence
		
		last_gesture = "fist"
		
		while self.running:
			ret, frame = self.cap.read()
			if not ret:
				break
			
			# Predict
			tensor = self.preprocess_frame(frame)
			with torch.no_grad():
				output = self.model(tensor)
				probs = torch.softmax(output, dim=1)
				pred = torch.argmax(probs, dim=1).item()
				conf = probs[0][pred].item()
			
			gesture = self.gesture_names[pred]
			current_gesture = gesture
			gesture_confidence = conf
			
			# CONTINUOUS JUMP: Peace held → keep jumping
			if gesture == "peace" and conf > 0.80:
				gesture_can_jump = True
				if last_gesture != "peace":
					last_gesture = "peace"
			else:
				# Not peace → stop jumping
				gesture_can_jump = False
				if last_gesture == "peace":
					last_gesture = "fist"
			
			# Display
			h, w = frame.shape[:2]
			size = min(h, w)
			start_h = (h - size) // 2
			start_w = (w - size) // 2
			color = (0, 255, 0) if gesture == "peace" else (255, 255, 255)
			cv2.rectangle(frame, (start_w, start_h), (start_w + size, start_h + size), color, 2)
			cv2.putText(frame, f"{gesture.upper()} {conf:.0%}", (20, 40),
			           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
			
			# Show status
			status = "JUMPING!" if gesture_can_jump else "Hold peace to jump"
			status_color = (0, 255, 0) if gesture_can_jump else (255, 255, 255)
			cv2.putText(frame, status, (20, h - 20),
			           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
			
			cv2.imshow('Gesture Detection', frame)
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		
		self.cap.release()
		cv2.destroyAllWindows()
	
	def start(self):
		"""Start detection in background thread"""
		self.running = True
		self.thread = threading.Thread(target=self.detect_loop, daemon=True)
		self.thread.start()
	
	def stop(self):
		"""Stop detection"""
		self.running = False


# Setup game objects
pipe_group = pygame.sprite.Group()
bird_group = pygame.sprite.Group()
flappy = Bird(100, int(screen_height / 2))
bird_group.add(flappy)
button = Button(screen_width // 2 - 50, screen_height // 2 - 100, button_img)


def main():
	"""Main game with gesture control"""
	global flying, game_over, score, ground_scroll, last_pipe, pass_pipe, gesture_can_jump
	
	# Start gesture detection
	try:
		detector = SimpleGestureDetector(model_path="gesture_model.pth")
		detector.start()
	except FileNotFoundError:
		if input("Model not found! Run without gestures? (y/n): ").lower() != 'y':
			return
		detector = None
	
	run = True
	frame_count = 0
	while run:
		clock.tick(fps)
		frame_count += 1
		
		# Draw background
		screen.blit(bg, (0, 0))
		
		# Draw and update
		pipe_group.draw(screen)
		bird_group.draw(screen)
		bird_group.update()
		screen.blit(ground_img, (ground_scroll, 768))
		
		# Score
		if len(pipe_group) > 0:
			if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.left \
			   and bird_group.sprites()[0].rect.right < pipe_group.sprites()[0].rect.right \
			   and pass_pipe == False:
				pass_pipe = True
			if pass_pipe == True:
				if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.right:
					score += 1
					pass_pipe = False
		
		draw_text(str(score), font, white, int(screen_width / 2), 20)
		draw_gesture_status()
		
		# Collision
		if pygame.sprite.groupcollide(bird_group, pipe_group, False, False) or flappy.rect.top < 0:
			game_over = True
		if flappy.rect.bottom >= 768:
			game_over = True
			flying = False
		
		# Game running
		if flying == True and game_over == False:
			time_now = pygame.time.get_ticks()
			if time_now - last_pipe > pipe_frequency:
				pipe_height = random.randint(-100, 100)
				btm_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, -1)
				top_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, 1)
				pipe_group.add(btm_pipe)
				pipe_group.add(top_pipe)
				last_pipe = time_now
			
			pipe_group.update()
			ground_scroll -= scroll_speed
			if abs(ground_scroll) > 35:
				ground_scroll = 0
		
		# Game over
		if game_over == True:
			if button.draw():
				game_over = False
				score = reset_game()
		
		# Start game - check gesture directly, not just the jump flag
		keys = pygame.key.get_pressed()
		if not flying and not game_over:
			# Allow peace sign (held or transition) or keyboard to start
			peace_ready = (current_gesture == "peace" and gesture_confidence > 0.80)  # Reduced threshold
			if keys[pygame.K_UP] or gesture_can_jump or peace_ready:
				flying = True
		
		# Events
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
			if event.type == pygame.MOUSEBUTTONDOWN:
				if not flying and not game_over:
					flying = True
		
		pygame.display.update()
	
	# Cleanup
	if detector:
		detector.stop()
	pygame.quit()


if __name__ == "__main__":
	main()
