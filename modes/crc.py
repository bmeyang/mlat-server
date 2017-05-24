# -*- mode: python; indent-tabs-mode: nil -*-

# Part of mlat-server: a Mode S multilateration server
# Copyright (C) 2015  Oliver Jowett <oliver@mutability.co.uk>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Calculates the 24-bit CRC used in Mode S messages.
"""


# on my system, the generic version is fractionally slower than an unrolled
# version, but the difference is eaten by the cost of having a wrapper to
# decide which version to use. So let's just do this the simple way.
def residual(payload):
    """Computes the 24-bit Mode S CRC residual for a message.

    The CRC residual is the CRC computed across the first 4 or 11 bytes,
    XOR-ed with the CRC value stored in the final 3 bytes.

    For a message using Address/Parity, the expected residual is the
    transmitter's address.

    For a message using Parity/Interrogator, the expected residual is
    the interrogator ID.

    For an extended squitter message or a DF11 acquisition squitter, the
    expected residual is zero.

    Errors in the message or in the CRC value itself will appear as errors
    in the residual value.
    """

    t = _crc_table
    rem = t[payload[0]]
    for b in payload[1:-3]:
        rem = ((rem & 0xFFFF) << 8) ^ t[b ^ (rem >> 16)]

    rem = rem ^ (payload[-3] << 16) ^ (payload[-2] << 8) ^ (payload[-1])
    return rem


def _make_table():
    # precompute the CRC table
    t = []

    poly = 0xfff409
    for i in range(256):
        c = i << 16
        for j in range(8):
            if c & 0x800000:
                c = (c << 1) ^ poly
            else:
                c = (c << 1)

        t.append(c & 0xffffff)

    return t

if __name__ == '__main__':
    _crc_table = _make_table()

    print('# -*- mode: python; indent-tabs-mode: nil -*-')
    print('# generated by modes.crc: python3 -m modes.crc')
    print()
    print('table = (')
    for i in range(0, 256, 8):
        print('    ' + ', '.join(['0x{0:06x}'.format(c) for c in _crc_table[i:i+8]]) + ',')
    print(')')
else:
    try:
        from .crc_lookup import table as _crc_table
    except ImportError:
        _crc_table = _make_table()
