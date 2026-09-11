"""Bonjour / DNS-SD advertisement, so the phone can find this PC.

``docs/BOOSTER_PROTOCOL.md`` section 2. We advertise; the phone browses with
``NWBrowser`` and shows every instance it finds.

Two details that decide whether this works at all:

* **The Bonjour instance name is what the user sees** in the device list before
  pairing, so it is named after the computer ("Ollie's PC"), never after the
  product.
* The client **does not read the TXT record today**. We publish ``api``,
  ``bid`` and ``name`` anyway - they cost nothing, and a later client will use
  them to show a paired device's real name and to reject an incompatible
  Booster before the user taps anything.
"""

from __future__ import annotations

import logging
import socket
from typing import List, Optional

from .. import brand
from ..protocol import wire

log = logging.getLogger(__name__)


def local_ip_addresses() -> List[str]:
    """Best-effort list of this machine's LAN IPv4 addresses.

    Used both for the Bonjour advertisement and to show the user a "your PC is
    at 192.168.1.42" line in the GUI, which is the fallback when Bonjour is
    blocked by a router or a corporate Wi-Fi profile.

    The UDP-connect trick finds the address that would actually be used to
    reach the LAN, which is the right one on a machine with a VPN adapter, a
    Docker bridge and three virtual NICs - ``gethostbyname`` on such a machine
    routinely returns a useless one.
    """
    addresses: List[str] = []
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packet is sent; connect() on UDP just picks a route.
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1, guaranteed unroutable
        addresses.append(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address not in addresses and not address.startswith("127."):
                addresses.append(address)
    except OSError:
        pass
    return addresses


class BonjourAdvertiser:
    """Registers ``_nimbusboost._tcp`` in ``local.`` for as long as it is open.

    ``zeroconf`` is a hard dependency of the server, but a machine with a
    firewall that blocks mDNS, or a second copy of the Booster already running,
    should degrade to "reachable by IP address" rather than refuse to start.
    Every failure here is logged and swallowed, and :attr:`last_error` carries
    the reason to the GUI so the user is told why discovery is not working.
    """

    def __init__(self, booster_id: str, name: str, port: int) -> None:
        self.booster_id = booster_id
        self.name = name
        self.port = port
        self.last_error: Optional[str] = None
        self._zeroconf = None
        self._info = None

    @property
    def is_advertising(self) -> bool:
        return self._info is not None

    def start(self) -> bool:
        """Register the service. Returns False (and logs) if mDNS is unusable."""
        if self._info is not None:
            return True
        try:
            from zeroconf import ServiceInfo, Zeroconf
        except ImportError as error:  # pragma: no cover - dependency missing
            self.last_error = (
                "The zeroconf package is not installed, so this PC cannot "
                "announce itself. The phone can still connect by IP address."
            )
            log.warning("zeroconf import failed: %s", error)
            return False

        addresses = []
        for text in local_ip_addresses():
            try:
                addresses.append(socket.inet_aton(text))
            except OSError:
                continue

        # The instance name is what the user reads in the phone's list.
        # zeroconf wants "<instance>.<type>.<domain>".
        instance = "{name}.{type}{domain}".format(
            name=_escape_instance_name(self.name),
            type=brand.BOOSTER_SERVICE_TYPE + ".",
            domain=brand.BOOSTER_SERVICE_DOMAIN,
        )
        try:
            self._zeroconf = Zeroconf()
            self._info = ServiceInfo(
                type_=brand.BOOSTER_SERVICE_TYPE + "." + brand.BOOSTER_SERVICE_DOMAIN,
                name=instance,
                port=int(self.port),
                addresses=addresses,
                properties={
                    # Keys are deliberately short: a TXT record is capped at
                    # 255 bytes per string and this has to survive a rename.
                    "api": wire.API_VERSION,
                    "bid": self.booster_id,
                    "name": self.name,
                },
                server=_dns_safe_hostname() + ".",
            )
            self._zeroconf.register_service(self._info)
            self.last_error = None
            log.info("Advertising %s on port %s", instance, self.port)
            return True
        except Exception as error:  # noqa: BLE001 - mDNS failure must not be fatal
            self.last_error = (
                "This PC could not announce itself on the network, so the "
                "phone may not find it automatically. Connecting by IP "
                "address still works."
            )
            # The exception TYPE is logged as well as its text, because
            # zeroconf raises several exceptions whose str() is empty, and
            # "Bonjour registration failed: " with nothing after it is the
            # least useful log line it is possible to write.
            log.warning(
                "Bonjour registration failed: %s: %s",
                type(error).__name__,
                error or "(no detail)",
            )
            self._shutdown_zeroconf()
            return False

    def update_name(self, name: str) -> None:
        """Rename the advertised instance: unregister, then register again."""
        if name == self.name:
            return
        self.name = name
        if self._info is not None:
            self.stop()
            self.start()

    def stop(self) -> None:
        if self._zeroconf is not None and self._info is not None:
            try:
                self._zeroconf.unregister_service(self._info)
            except Exception as error:  # noqa: BLE001
                log.debug("unregister_service failed: %s", error)
        self._info = None
        self._shutdown_zeroconf()

    def _shutdown_zeroconf(self) -> None:
        if self._zeroconf is not None:
            try:
                self._zeroconf.close()
            except Exception as error:  # noqa: BLE001
                log.debug("Zeroconf.close failed: %s", error)
        self._zeroconf = None


def _escape_instance_name(name: str) -> str:
    """Make a computer name safe as a DNS-SD instance label.

    Instance names are allowed to be rich UTF-8, but a dot inside one splits
    the name into extra labels and produces a service nobody can resolve.
    """
    cleaned = name.replace(".", " ").strip()
    return (cleaned or "PC")[:63]


def _dns_safe_hostname() -> str:
    """``<host>.local`` with anything DNS dislikes replaced."""
    try:
        host = socket.gethostname().split(".")[0]
    except OSError:
        host = "booster"
    safe = "".join(c if (c.isalnum() or c == "-") else "-" for c in host)
    safe = safe.strip("-") or "booster"
    return safe[:63] + "." + brand.BOOSTER_SERVICE_DOMAIN.rstrip(".")
