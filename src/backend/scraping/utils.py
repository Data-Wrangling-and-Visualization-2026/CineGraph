import os
from typing import Generator

from settings import settings


def create_data_folder() -> None:
    """
    Check if /data folder is present. If no then creates it.
    """
    os.makedirs(
        settings.scraper.output_folder,
        exist_ok=True
    )


def collect_proxy_ips() -> list[str]:
    """
    Collects list of proxies from env.

    Returns:
        list[str]: List of proxies in format 'ip:port'
    """
    current_ip_n = 1
    ips = []

    port = os.getenv('PROXY_PORT') # one port for all IPs
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
    """
    Creates generator for proxies. Uses Round-Robin alg.

    Returns:
        None: if no proxies are present

    Yields:
        Generator[str, None, None]: generator
    """
    ips = collect_proxy_ips()

    if not len(ips):
        print('No available IPs!')
        while True:
            yield None

    i = 0
    while True:
        yield ips[i]
        i = (i + 1) % len(ips)
