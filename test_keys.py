import sys, select, termios, tty

print("Press any key (W/A/S/D). Press 'q' to quit.")
settings = termios.tcgetattr(sys.stdin)
try:
    while True:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            print(f"\rKey Pressed: '{key}'             ", end="")
            if key == 'q':
                break
        else:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    print("\nDone.")