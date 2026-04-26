#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
parse_sni.py — Extract SNI hostnames from a pcap.

Reads a pcap file written by tcpdump, finds every TLS ClientHello, and
writes a CSV of (timestamp, sni) rows. Pure stdlib — no scapy/dpkt.

Compatible with Python 2.7 (Ubuntu 12.04 default).

Usage:
    python parse_sni.py <input.pcap> <output.csv>

Pcap format reference: https://wiki.wireshark.org/Development/LibpcapFileFormat
TLS ClientHello reference: RFC 5246 (TLS 1.2), RFC 6066 (SNI extension)
"""

from __future__ import print_function
import struct
import sys
import csv

# --- Pcap format constants --------------------------------------------

PCAP_MAGIC_LE = 0xa1b2c3d4   # little-endian
PCAP_MAGIC_BE = 0xd4c3b2a1   # big-endian (machine wrote it big-endian)
PCAP_GLOBAL_HEADER_LEN = 24
PCAP_RECORD_HEADER_LEN = 16

# Link-layer types we handle
LINKTYPE_ETHERNET = 1
LINKTYPE_LINUX_SLL = 113   # tcpdump on Linux "any" interface uses this

# --- TLS / SNI parsing ------------------------------------------------

def parse_client_hello_sni(payload):
    """
    Given the TCP payload of one packet, return SNI hostname if this is a
    TLS ClientHello with a server_name extension; otherwise None.

    TLS record:    [type(1)] [version(2)] [length(2)] [data...]
                   type 0x16 = handshake
    Handshake:     [type(1)] [length(3)] [data...]
                   type 0x01 = ClientHello
    ClientHello:   [version(2)] [random(32)] [session_id_len(1)] [session_id]
                   [cipher_suites_len(2)] [cipher_suites]
                   [compression_methods_len(1)] [compression_methods]
                   [extensions_len(2)] [extensions...]
    Extension:     [type(2)] [length(2)] [data]
                   type 0x0000 = server_name
    server_name:   [list_len(2)] [name_type(1)] [name_len(2)] [hostname]
                   name_type 0x00 = host_name
    """
    try:
        if len(payload) < 5:
            return None
        # TLS record header
        if payload[0:1] != b'\x16':            # not handshake
            return None
        rec_len = struct.unpack('!H', payload[3:5])[0]
        body = payload[5:5 + rec_len]
        if len(body) < 4:
            return None
        # Handshake header
        if body[0:1] != b'\x01':               # not ClientHello
            return None
        # Skip handshake_type(1) + handshake_length(3) + version(2) + random(32) = 38
        i = 38
        # session_id
        sid_len = ord(body[i:i+1])
        i += 1 + sid_len
        # cipher_suites
        cs_len = struct.unpack('!H', body[i:i+2])[0]
        i += 2 + cs_len
        # compression_methods
        cm_len = ord(body[i:i+1])
        i += 1 + cm_len
        # extensions_length
        if i + 2 > len(body):
            return None
        ext_total = struct.unpack('!H', body[i:i+2])[0]
        i += 2
        end = i + ext_total
        # walk extensions
        while i + 4 <= end:
            ext_type, ext_len = struct.unpack('!HH', body[i:i+4])
            i += 4
            if ext_type == 0x0000:             # server_name
                # server_name list: list_len(2), name_type(1), name_len(2), name
                # i points at start of extension data
                sn_list_len = struct.unpack('!H', body[i:i+2])[0]
                name_type = ord(body[i+2:i+3])
                if name_type != 0x00:
                    return None
                name_len = struct.unpack('!H', body[i+3:i+5])[0]
                hostname = body[i+5:i+5+name_len]
                return hostname.decode('ascii', errors='replace')
            i += ext_len
        return None
    except (struct.error, IndexError):
        return None

# --- Pcap reader ------------------------------------------------------

def read_pcap(path):
    """Yield (timestamp_float, link_type, packet_bytes) for each record."""
    with open(path, 'rb') as f:
        gh = f.read(PCAP_GLOBAL_HEADER_LEN)
        if len(gh) < PCAP_GLOBAL_HEADER_LEN:
            return
        magic = struct.unpack('<I', gh[0:4])[0]
        if magic == PCAP_MAGIC_LE:
            endian = '<'
        elif magic == PCAP_MAGIC_BE:
            endian = '>'
        else:
            raise ValueError("Not a pcap file (bad magic: 0x%x)" % magic)
        link_type = struct.unpack(endian + 'I', gh[20:24])[0]

        while True:
            rh = f.read(PCAP_RECORD_HEADER_LEN)
            if len(rh) < PCAP_RECORD_HEADER_LEN:
                return
            ts_sec, ts_usec, incl_len, _orig_len = struct.unpack(
                endian + 'IIII', rh)
            data = f.read(incl_len)
            if len(data) < incl_len:
                return
            yield (ts_sec + ts_usec / 1e6, link_type, data)

# --- Layer stripping --------------------------------------------------

def extract_tcp_payload(link_type, frame):
    """Strip link + IP + TCP headers, return TCP payload bytes or None."""
    try:
        if link_type == LINKTYPE_ETHERNET:
            if len(frame) < 14:
                return None
            ethertype = struct.unpack('!H', frame[12:14])[0]
            if ethertype != 0x0800:            # not IPv4
                return None
            ip_offset = 14
        elif link_type == LINKTYPE_LINUX_SLL:
            # SLL header is 16 bytes; protocol field at offset 14
            if len(frame) < 16:
                return None
            proto = struct.unpack('!H', frame[14:16])[0]
            if proto != 0x0800:
                return None
            ip_offset = 16
        else:
            return None

        ip = frame[ip_offset:]
        if len(ip) < 20:
            return None
        ihl = (ord(ip[0:1]) & 0x0f) * 4        # IP header length in bytes
        protocol = ord(ip[9:10])
        if protocol != 6:                      # not TCP
            return None
        tcp = ip[ihl:]
        if len(tcp) < 20:
            return None
        data_offset = (ord(tcp[12:13]) >> 4) * 4
        return tcp[data_offset:]
    except (struct.error, IndexError):
        return None

# --- Main -------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python parse_sni.py <input.pcap> <output.csv>",
              file=sys.stderr)
        sys.exit(1)

    pcap_path, csv_path = sys.argv[1], sys.argv[2]
    count = 0

    with open(csv_path, 'wb') as out:
        w = csv.writer(out)
        w.writerow(['timestamp', 'sni'])
        for ts, link_type, frame in read_pcap(pcap_path):
            payload = extract_tcp_payload(link_type, frame)
            if not payload:
                continue
            sni = parse_client_hello_sni(payload)
            if sni:
                w.writerow(['%.6f' % ts, sni])
                count += 1

    print("Parsed %d SNIs from %s" % (count, pcap_path))

if __name__ == '__main__':
    main()