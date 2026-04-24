#!/bin/bash
# Runs on the attacker VM: uses tshark to sniff TLS Client Hello packets from the victim IP
# and writes captured SNI hostnames and timestamps to data/captured_sni.csv
