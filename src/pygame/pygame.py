import pygame

SCREEN_W = 1000
SCREEN_H = 800

def setup_pygame():

    pygame.init()

    pygame.display.set_caption("Love Briscola")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    tempo = [pygame.time.get_ticks()]
    message = ""
    return screen, tempo, message

def text(txt, posit, color=(0, 0, 0), size=40):
    pos = (SCREEN_W // 2 + posit[0], posit[1])
    font = pygame.font.Font("Pixeltype.ttf", size)
    txtsurf = font.render(txt, False, color)
    text_rect = txtsurf.get_rect(midtop=pos)
    self.screen.blit(txtsurf, text_rect)