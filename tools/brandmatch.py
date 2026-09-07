"""Fail if the phone and the PC Booster disagree about the brand.

WHY THIS EXISTS. The app was renamed from Nimbus3D to LiKOVA by editing the
brand block in ios/project.yml, which is a single source of truth for the SWIFT
side and only for the Swift side. The Python Booster carries its own four brand
values, and nothing connected the two.

So the phone started browsing for `_likovaboost._tcp` while the Booster went on
registering `_nimbusboost._tcp`, and the two could never see each other again.
The owner found it the only way left: "it's not picking up the booster". The
existing booster-check job compiles the Python and runs pyflakes, and neither
of those can notice that two constants in two languages have drifted apart.

The DNS-SD service type is the one that silently breaks discovery, but the
documents folder and the display name are checked too, because a mismatch there
means the Booster writes to a folder the phone never mentions and calls itself
by a name the user does not recognise.

Read as data, never imported: this parses both files with string matching
rather than importing the Python package, so it needs no dependencies and
cannot be fooled by an import side effect.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJECT = os.path.join(ROOT, 'ios', 'project.yml')
BRAND = os.path.join(ROOT, 'booster', 'src', 'nimbus_booster', 'brand.py')


def yaml_brand_value(text, key):
    """The value of one key in the settingGroups.brand block."""
    match = re.search(
        r'^\s*' + re.escape(key) + r'\s*:\s*(.+?)\s*$', text, re.MULTILINE
    )
    if not match:
        return None
    value = match.group(1).strip()
    if value and value[0] in '"\'' and value[-1] == value[0]:
        value = value[1:-1]
    return value


def python_brand_value(text, name):
    match = re.search(
        r'^' + re.escape(name) + r'\s*=\s*"([^"]*)"\s*$', text, re.MULTILINE
    )
    return match.group(1) if match else None


def main():
    project = io.open(PROJECT, encoding='utf-8').read()
    brand = io.open(BRAND, encoding='utf-8').read()

    checks = [
        ('brand slug',
         yaml_brand_value(project, 'NIMBUS_BRAND_SLUG'),
         python_brand_value(brand, 'SLUG'),
         'the DNS-SD service type is built from this, so a mismatch means the '
         'phone and the Booster can never discover each other'),
        ('display name',
         yaml_brand_value(project, 'NIMBUS_DISPLAY_NAME'),
         python_brand_value(brand, 'DISPLAY_NAME'),
         'the Booster would call itself by a name the user does not recognise'),
        ('documents folder',
         yaml_brand_value(project, 'NIMBUS_DOCS_FOLDER'),
         python_brand_value(brand, 'DOCUMENTS_FOLDER_NAME'),
         'the Booster would write scans to a folder the phone never mentions'),
        ('bundle identifier',
         yaml_brand_value(project, 'NIMBUS_BUNDLE_ID'),
         python_brand_value(brand, 'BUNDLE_IDENTIFIER'),
         'informational on the Booster, but a mismatch means one of the two '
         'was renamed and the other was forgotten'),
    ]

    bad = []
    print('brandmatch: ios/project.yml against booster brand.py')
    for label, phone, pc, why in checks:
        if phone is None or pc is None:
            bad.append((label, phone, pc, 'could not be read from one side'))
            print('  %-18s UNREADABLE  phone=%r  booster=%r' % (label, phone, pc))
        elif phone != pc:
            bad.append((label, phone, pc, why))
            print('  %-18s MISMATCH    phone=%r  booster=%r' % (label, phone, pc))
        else:
            print('  %-18s ok          %s' % (label, phone))

    if not bad:
        print()
        print('PASS: the phone and the Booster agree about who they are.')
        return 0

    print()
    print('FAIL: the phone and the PC Booster have drifted apart.')
    print('Renaming the product means editing BOTH the brand block in')
    print('ios/project.yml AND the four values at the top of')
    print('booster/src/nimbus_booster/brand.py.')
    print()
    for label, phone, pc, why in bad:
        print('  %s: phone says %r, booster says %r' % (label, phone, pc))
        print('      %s' % why)
    return 1


if __name__ == '__main__':
    sys.exit(main())
