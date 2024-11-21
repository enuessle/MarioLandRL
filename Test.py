import numpy as np
from pyboy import PyBoy
import keyboard

supermarioland_rom = "SuperMarioLand_rom.gb"

# Map keyboard keys to actions (define these according to your game controls)
actions_map = {
    'up': 'UP',
    'down': 'DOWN',
    'left': 'LEFT',
    'right': 'RIGHT',
    'space': 'A',  # Assume space is jumping
    'enter': 'START'
}

if __name__ == '__main__':
    pyboy = PyBoy(supermarioland_rom)
    pyboy.set_emulation_speed(1)
    assert pyboy.cartridge_title == "SUPER MARIOLAN"
    mario = pyboy.game_wrapper
    mario.game_area_mapping(mario.mapping_compressed, 0)
    mario.start_game()

    while True:
        # Check for keyboard input and map it to PyBoy actions
        if keyboard.is_pressed('up'):
            pyboy.button('UP')
        if keyboard.is_pressed('down'):
            pyboy.button('DOWN')
        if keyboard.is_pressed('left'):
            pyboy.button('LEFT')
        if keyboard.is_pressed('right'):
            pyboy.button('RIGHT')
        if keyboard.is_pressed('space'):
            pyboy.button('A')  # Jumping action (or other game action)
        if keyboard.is_pressed('enter'):
            pyboy.button('START')  # Start button or pause

        # Step the emulator forward by 1 tick (or frame)
        pyboy.tick()
        print(f"Score: {mario.level_progress}")