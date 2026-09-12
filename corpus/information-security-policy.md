---
doc_id: POL-SEC-001
title: Information Security and Acceptable Use Policy
version: 6.2
effective_date: 2026-01-01
owner: Information Security
applies_to: All employees, contractors, and interns
related: POL-REMOTE-001, POL-INTL-001, POL-EQUIP-001
---

# Information Security and Acceptable Use Policy

## 1. Purpose

This policy defines how Helios Systems information and systems must be handled.
It applies to all workers — employees, contractors, and interns — and to all
devices used for Helios work, in any location.

## 2. Data Classification

All Helios information falls into one of four classifications. The
classification determines the handling requirements.

| Class | Definition | Examples | Handling |
| --- | --- | --- | --- |
| **Public** | Approved for external release | Marketing site, published docs | No restriction |
| **Internal** | Default for business information | Org charts, roadmaps, meeting notes | Helios accounts only |
| **Confidential** | Would harm Helios or a customer if disclosed | Customer data, contracts, source code, unreleased financials | Encrypted at rest and in transit; access on a need-to-know basis |
| **Restricted** | Legally protected or highly sensitive | Employee PII, health data, payroll, security keys, M&A material | Named-individual access only; access logged; never on personal devices |

When a classification is unclear, treat the information as **Confidential**
until Information Security advises otherwise.

## 3. Devices

### 3.1 Company-Issued Devices

All Confidential and Restricted work must be performed on a Helios-issued
device. Helios-issued laptops are centrally managed and must have:

- full-disk encryption enabled (enforced by MDM);
- the Helios endpoint-detection agent running;
- automatic OS security updates enabled, with critical patches applied within
  **14 days** of release;
- screen lock after no more than **10 minutes** of inactivity; and
- no unapproved administrative software.

### 3.2 Personal Devices (BYOD)

Personal devices may be used **only** for: Helios email, calendar, and the
Helios chat client, each through an approved managed application. Personal
devices must have a device passcode and current OS version enrolled in the
Helios MDM.

Personal devices may **never** be used to access Restricted data, to store
Helios source code, or to perform any work while outside the home country
(see `POL-INTL-001` Section 6).

### 3.3 Lost or Stolen Devices

Report a lost or stolen device to the Security Operations Center immediately,
and in any case within **4 hours** of discovery, at
`security@helios.example.com` or +1-555-0100. Helios will remotely wipe the
device. Late reporting is itself a policy violation.

## 4. Workspace and Physical Security

Employees working outside a Helios office must:

- hold calls involving Confidential or Restricted information where they cannot
  be overheard — not in coworking common areas, cafés, aircraft, or trains;
- use a privacy screen when working on Confidential material in any public
  space;
- never leave a device unattended and unlocked; and
- shred or securely dispose of printed Confidential material.

Video calls involving Restricted data require a private room and a blurred or
virtual background.

## 5. Authentication and Access

- Multi-factor authentication is mandatory for all Helios systems. SMS-based
  MFA is not accepted; use the Helios authenticator app or a hardware key.
- Passwords must be unique per system and generated and stored in the
  company-provided password manager.
- Sharing credentials is prohibited without exception, including with a manager
  or with IT support.
- Access is granted on a least-privilege basis and is reviewed quarterly.
- Report suspected credential compromise within **1 hour** of discovery.

## 6. Acceptable Use

Helios systems may be used for incidental personal purposes that do not
interfere with work, consume significant resources, or violate this policy.

The following are prohibited:

- installing unapproved software, browser extensions, or AI tools that transmit
  Helios data to third parties;
- pasting Confidential or Restricted data into any external AI service not on
  the Helios approved-tools list;
- connecting Helios devices to untrusted USB peripherals or charging stations;
- circumventing security controls, including VPN split-tunneling, MDM removal,
  or disabling the endpoint agent;
- using Helios systems for outside commercial activity; and
- accessing data not required for the employee's role.

## 7. Travel and Remote-Location Security

### 7.1 Security Review Triggers

An Information Security review is required before:

- any work from outside the employee's home country, of any duration
  (`POL-INTL-001` Section 4.5);
- domestic temporary work exceeding **10 consecutive business days**
  (`POL-REMOTE-001` Section 5); and
- any travel to a country on the Elevated Risk list (Section 8).

The review is requested in the same Workday flow as the travel or remote-work
request and typically completes within **5 business days**.

### 7.2 Requirements While Travelling

- Connect through the Helios VPN for **all** work activity while outside the
  registered work location. Public Wi-Fi without VPN is prohibited.
- Use a company eSIM or personal hotspot in preference to hotel or café Wi-Fi.
- Enable device-location services so a lost device can be traced.
- Do not connect to Helios systems from a device you do not control, including
  hotel business-centre computers.
- Physical device custody must be maintained at borders; do not place Helios
  devices in checked baggage.

### 7.3 Loaner Devices

Travel to an Elevated Risk country requires a clean loaner laptop and loaner
phone, issued by IT with at least **5 business days'** notice. The employee's
primary device must not travel. Loaner devices are wiped on return.

## 8. Elevated Risk Locations

The following locations require a loaner device, pre-travel briefing, and
post-travel device wipe: China, Russia, Belarus, Iran, North Korea, Syria, and
Cuba. Work from sanctioned jurisdictions is prohibited regardless of device.

The Elevated Risk list is maintained by Information Security and may change; the
current list is authoritative at the time of travel.

## 9. Incident Reporting

Report any suspected security incident — phishing, data exposure, malware,
unauthorised access, or lost device — to `security@helios.example.com`
immediately. Helios does not penalise good-faith reports, including reports of
an employee's own mistake. Failing to report a known incident is a serious
violation.

## 10. Enforcement

Violations may result in revocation of access, disciplinary action up to and
including termination, and, where applicable, legal action. Suspected criminal
activity is referred to law enforcement.

## 11. Contacts

Information Security: `security@helios.example.com`
Security Operations Center (24/7): +1-555-0100
