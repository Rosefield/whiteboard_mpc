F = GF(2**128, name='a', modulus=x^128 + x^7 + x^2 + x + 1)

def split(x):
    return (x.integer_representation() >> 64, x.integer_representation() & (2**64 - 1))

def output_mul(x,y,z):
    return f"""
    {{
        let x = FF2_128::new{split(x)};
        let y = FF2_128::new{split(y)};
        let z = FF2_128::new{split(z)};
        assert!(x*y == z);
    }}
    """

def make_mul_case():
    x,y = F.random_element(), F.random_element()
    z = x*y
    print(output_mul(x,y,z))

make_mul_case()
make_mul_case()
make_mul_case()


def output_inv(x,y):
    return f"""
    {{
        let x = FF2_128::new{split(x)};
        let y = FF2_128::new{split(y)};
        assert!(x.inv() == Some(y));
    }}
    """

def make_inv_case():
    x = F.random_element()
    y = x**-1
    print(output_inv(x,y))

make_inv_case()
make_inv_case()
make_inv_case()
