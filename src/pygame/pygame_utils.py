from numpy import array
import pygame

main_menu_sprite_group = pygame.sprite.Group()


class Button(pygame.sprite.Sprite):
    def __init__(self, width, height, background_color, position, text_size, text_content, text_color):
        super().__init__(main_menu_sprite_group)
        self.image = pygame.Surface([width, height])
        self.image.fill(background_color)
        self.rect = self.image.get_rect(center=position)
        self.text = pygame.font.Font("src/asset/Pixel-type.ttf", text_size).render(text_content, True, text_color)
        self.text_rect = self.text.get_rect(center=[width/2, height/2])
        self.image.blit(self.text, self.text_rect)

    def is_pressed(self, mouse_pos, pressed):
        if self.rect.collidepoint(mouse_pos) and pressed[0]:
            return True
        return False

