import torch


def rotate_reference(x, theta):
    # Naive and readable implementation
    assert x.shape[0] % 2 == 0
    half = x.shape[0] // 2

    sin = theta.sin()
    cos = theta.cos()

    point_x = x[:half]
    point_y = x[half:]
    rotated_point_x = point_x * cos - point_y * sin
    rotated_point_y = point_x * sin + point_y * cos
    output = torch.cat([rotated_point_x, rotated_point_y], dim=-1)

    return output


def rotate_with_rotate_half(x, theta):
    # Commonly used implementation
    assert x.shape[0] % 2 == 0

    theta_repeated = torch.cat([theta, theta], dim=-1)
    cos = theta_repeated.cos()
    sin = theta_repeated.sin()

    def rotate_half(x):
        half = x.shape[0] // 2
        x1 = x[:half]
        x2 = x[half:]
        return torch.cat((-x2, x1), dim=-1)

    output = x * cos + rotate_half(x) * sin

    return output


def rotate_with_rotate_half_inplace(x, theta):
    # If you look properly, you'll see that it's the same as `rotate_reference`.
    # And this implementation is faster then `rotate_with_rotate_half` in eager mode.
    assert x.shape[0] % 2 == 0
    half = x.shape[0] // 2

    cos = theta.cos()
    sin = theta.sin()

    output = torch.empty_like(x)
    x1 = x[:half]
    x2 = x[half:]
    output[:half] = x1 * cos - x2 * sin
    output[half:] = x2 * cos + x1 * sin

    return output


def transpose_wrapper(f):
    # A wrapper for converting between halving and odd-even splitting

    def _wrapper(*args, **kwargs):
        x, theta = args
        x_transposed = x.reshape((2, -1)).permute(1, 0).reshape(-1).contiguous()
        output_transposed = f(x_transposed, theta)
        output = output_transposed.reshape(-1, 2).permute(1, 0).reshape((-1,)).contiguous()
        return output

    return _wrapper


@transpose_wrapper
def rotate_with_complex(x, theta):
    # Complex number implementation
    # If optimized, this is probably the fastest option in eager mode, but it doesn't support `torch.comple`.
    theta_i = torch.polar(torch.ones_like(theta), theta)
    x_i = torch.view_as_complex(x.reshape(-1, 2))
    output = torch.view_as_real(x_i * theta_i).reshape(-1)

    return output


def main():
    N = 16
    assert N % 2 == 0
    x = torch.linspace(0, 1, N)
    theta = torch.linspace(0, torch.pi * 2, N // 2)

    z1 = rotate_reference(x, theta)
    print("rotate_reference\n", z1)
    z2 = rotate_with_rotate_half(x, theta)
    print("rotate_with_rotate_half\n", z2)
    z3 = rotate_with_rotate_half_inplace(x, theta)
    print("rotate_with_rotate_half_inplace\n", z3)
    z4 = rotate_with_complex(x, theta)
    print("rotate_with_complex\n", z4)

    check1 = torch.all(torch.isclose(z1, z2))
    print("rotate_reference == rotate_with_rotate_half", check1)
    assert check1

    check2 = torch.all(torch.isclose(z1, z3))
    print("rotate_reference == rotate_with_rotate_half_inplace", check2)
    assert check2

    check3 = torch.all(torch.isclose(z1, z4))
    print("rotate_reference == rotate_with_complex", check3)
    assert check3


if __name__ == "__main__":
    main()
