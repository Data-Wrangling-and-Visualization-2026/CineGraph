import os
from typing import Generator

from settings import settings


def create_data_folder() -> None:
    os.makedirs(
        settings.scraper.output_folder,
        exist_ok=True
    )


def collect_proxy_ips() -> list[str]:
    current_ip_n = 1
    ips = []

    port = os.getenv('PROXY_PORT')
    if not port:
        return ips

    while True:
        addr = os.getenv(f'IP_{current_ip_n}')

        if addr:
            full_addr = addr + ':' + port
            ips.append(full_addr)
        else:
            return ips

        current_ip_n += 1


def get_next_proxy() -> Generator[str, None, None]:
    ips = collect_proxy_ips()

    if not len(ips):
        print('No available IPs!')
        return None

    i = 0
    while True:
        yield ips[i]
        i = (i + 1) % len(ips)
