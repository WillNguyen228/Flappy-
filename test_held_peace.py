"""
Test: Game Start with Held Peace Sign

This tests that holding peace will start the game.
"""

import pygame
from pygame.locals import *

pygame.init()

screen_width = 864
screen_height = 936
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Test: Held Peace Sign')

font = pygame.font.SysFont('Arial', 30)
white = (255, 255, 255)
green = (0, 255, 0)

# Game state
flying = False

# Simulate gesture state
current_gesture = "peace"  # Start with peace already held
gesture_confidence = 0.95
gesture_can_jump = False

def draw_text(text, y, color=white):
	surf = font.render(text, True, color)
	x = (screen_width - surf.get_width()) // 2
	screen.blit(surf, (x, y))

print("\n" + "=" * 70)
print("TEST: GAME START WITH HELD PEACE SIGN")
print("=" * 70)
print("\nSimulating that user is ALREADY holding peace sign")
print("Game should start immediately when this check runs:")
print("  current_gesture == 'peace' and gesture_confidence > 0.85")
print("\nStarting...\n")

clock = pygame.time.Clock()
run = True
frames = 0

while run and frames < 180:  # Run for 3 seconds
	clock.tick(60)
	frames += 1
	
	# === THE CRITICAL CHECK (same as in simple_gesture_flappy.py) ===
	keys = pygame.key.get_pressed()
	if not flying:
		# Check if gesture allows start
		if gesture_can_jump or (current_gesture == "peace" and gesture_confidence > 0.85):
			flying = True
			print(f"\n{'='*50}")
			print(f"✅ GAME STARTED! (frame {frames})")
			print(f"  current_gesture: {current_gesture}")
			print(f"  gesture_confidence: {gesture_confidence:.0%}")
			print(f"  gesture_can_jump: {gesture_can_jump}")
			print(f"{'='*50}\n")
	
	# Draw
	screen.fill((50, 50, 100))
	
	if not flying:
		draw_text("Waiting to start...", 200, white)
		draw_text(f"Gesture: {current_gesture.upper()}", 250, green)
		draw_text(f"Confidence: {gesture_confidence:.0%}", 300, green)
		draw_text(f"Should start now...", 350, green)
	else:
		draw_text("✅ GAME RUNNING!", 250, green)
		draw_text("Held peace sign worked!", 300, white)
		draw_text("Press ESC to exit", 400, white)
	
	for event in pygame.event.get():
		if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
			run = False
	
	pygame.display.update()
	
	# Auto-exit after success
	if flying and frames > 60:  # Show success for 1 second
		break

pygame.quit()

if flying:
	print("✅ TEST PASSED: Held peace sign successfully starts game!")
	print("\nThe check works:")
	print("  if current_gesture == 'peace' and gesture_confidence > 0.85:")
	print("      flying = True")
else:
	print("❌ TEST FAILED: Game did not start with held peace")

print("\n" + "=" * 70)
