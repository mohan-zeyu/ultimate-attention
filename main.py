import torch
import test

def main():
    stdin = int(input("which one do you want to use?\n \
                0 for play\n \
                1 for calculation"))
    device_stdin = int(input("which device do you want to use?\n \
                0 for cpu\n \
                1 for gpu"))
    device = torch.device('mps') if device_stdin == 1 \
                            else torch.device('cpu')

    if stdin == 0:
        test.test_play()
    elif stdin == 1:
        test.test_kv_cache(device)
        

if __name__ == '__main__':
    main()
