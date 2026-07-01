# EXPORT CONTROL AND DUAL-USE TECHNOLOGY NOTICE

---

**Project:** RF Communication Resilience Enhancement for UAV Missions  
**Author:** Parth Sidhu (parthecstasy@gmail.com)  
**Effective Date:** July 2026

---

## IMPORTANT — READ BEFORE DOWNLOADING OR USING THIS REPOSITORY

This repository contains algorithms, simulation code, and research findings
relating to **electronic warfare (EW), adaptive beamforming, radio frequency
(RF) interference suppression, and unmanned aerial vehicle (UAV) resilience**.
These subject areas are classified as **dual-use technologies** under multiple
international and national export control frameworks.

By accessing, downloading, cloning, forking, or otherwise obtaining any part
of this repository, you represent and warrant that your access and use is
fully compliant with all applicable export control laws and regulations in
your jurisdiction.

---

> **⚠️ GEOPOLITICAL CONTEXT — INDIA, CHINA, PAKISTAN, AND BANGLADESH**
>
> This Work was authored by an Indian national. Its subject matter —
> UAV electronic warfare, adaptive beamforming, and RF jamming suppression
> — is directly relevant to defence domains in which the Republic of India
> maintains active and historically sensitive geopolitical relationships with
> the People's Republic of China, the Islamic Republic of Pakistan, and the
> People's Republic of Bangladesh.
>
> Users located in, or nationals of, these three countries are subject to
> **dedicated jurisdiction-specific legal notices** in addition to this
> general export control notice. See:
>
> - **China:** [`NOTICE_CHINA.md`](NOTICE_CHINA.md) — covers China's Export Control Law
>   (2020), Cryptography Law, Cybersecurity Law, Data Security Law, and
>   National Security Law. Provided in English and 中文.
>
> - **Pakistan & Bangladesh:** [`NOTICE_PAKISTAN_BANGLADESH.md`](NOTICE_PAKISTAN_BANGLADESH.md)
>   — covers Pakistan's Export Control Act 2004, PECA 2016, and Official
>   Secrets Act; and Bangladesh's Cyber Security Act 2023, Customs Act, and
>   Official Secrets Act. Provided in English, اردو, and বাংলা.
>
> **Reading this general notice alone is insufficient for users in these
> three jurisdictions. You must also read your dedicated notice in full.**

---

## SECTION 1 — NATURE OF THE TECHNOLOGY

The research contained herein falls within the following dual-use technology
categories recognised by major international control regimes:

- **Adaptive antenna systems and phased array beamforming** — relevant to
  Wassenaar Arrangement Category 3 (Electronics) and Category 5 Part 1
  (Telecommunications and Information Security).
- **Electronic warfare signal processing** — including jamming suppression,
  SINR optimisation, and spatial null steering.
- **Unmanned aerial vehicle (UAV) communications resilience** — command and
  control link protection for aerial platforms.
- **Neural network surrogate models for real-time RF physics** — AI-driven
  systems capable of sub-millisecond inference for real-time EW applications.

---

## SECTION 2 — SIMULATION-ONLY STATUS AND APPLICABILITY

This project is **purely a software simulation** implemented in Python on
a general-purpose personal computer. No hardware was fabricated, no signals
were transmitted, and no real UAV system was operated. See `DISCLAIMER.md`
for the full simulation-only notice.

The Author's position is that this Work, in its current form as academic
simulation source code, does not constitute a controlled "item" under most
major export control regimes, for the following reasons:

  a) The algorithms implemented are derived entirely from publicly available
     academic literature (cited in the project report). No classified,
     restricted, or government-furnished technical data was used.

  b) The software does not constitute a complete, operational system. It
     cannot be directly integrated into a deployed UAV platform without
     significant additional hardware development, calibration, and
     airworthiness certification.

  c) The ScanEagle airframe geometry used is a community-made 3D printable
     model downloaded from a public platform, not controlled technical data.

However, the Author makes **no legal representation** as to the export
control classification of this Work under any specific jurisdiction's laws.
Users are solely responsible for making their own export control
determinations before downloading or using this Work.

---

## SECTION 3 — WASSENAAR ARRANGEMENT

The Wassenaar Arrangement on Export Controls for Conventional Arms and
Dual-Use Goods and Technologies is a multilateral regime with 42
participating states. India is **not** a Wassenaar participating state,
however many countries that may access this repository are.

Relevant Wassenaar control list categories that *may* apply to concepts
in this research include:

- **Category 3 — Electronics:** Adaptive antenna systems, phased arrays.
- **Category 5 Part 1 — Telecommunications:** Spread spectrum systems,
  adaptive beamforming for communications.
- **Category 5 Part 2 — Information Security:** Where neural inference
  systems interface with protected communications channels.

Users in Wassenaar participating states are advised to consult their
national competent authority to determine whether downloading or using
this code requires an export licence.

---

## SECTION 4 — UNITED STATES (ITAR AND EAR)

**ITAR (International Traffic in Arms Regulations, 22 C.F.R. Parts 120–130)**

The Author is an Indian national residing in India. This Work was created
entirely in India using publicly available information. It is not derived
from any US Government contract, grant, or programme. The Author has not
accessed any ITAR-controlled technical data.

**However:** The ScanEagle UAV is a US-manufactured defence article
(USML Category VIII). Any US person or entity that accesses this
repository and then uses it in connection with actual ScanEagle hardware,
specifications, or classified technical data should consult a licensed
US export control attorney before doing so.

**EAR (Export Administration Regulations, 15 C.F.R. Parts 730–774)**

Publicly available academic research that does not involve Government
funding or classified data is generally covered by the EAR's "fundamental
research exclusion" (15 C.F.R. § 734.8). The Author believes this Work
qualifies for that exclusion. However, this is not a legal opinion, and
US persons should make their own determination.

---

## SECTION 5 — INDIA (SCOMET)

India's export control framework for dual-use items is the **SCOMET
(Special Chemicals, Organisms, Materials, Equipment and Technologies)**
list, administered under the Foreign Trade Policy by the Directorate
General of Foreign Trade (DGFT).

Electronic warfare signal processing systems and advanced antenna
technologies may fall under **SCOMET Category 5 (Electronics, Computers)**
or **Category 6 (Sensors and Lasers)**. The Author has not sought a formal
SCOMET classification opinion on this Work.

As a student researcher publishing academic findings, the Author believes
this Work is not subject to SCOMET export controls in its current form.
Any commercial entity in India seeking to use, adapt, or deploy concepts
from this Work should seek a formal SCOMET determination from DGFT.

---

## SECTION 6 — EUROPEAN UNION (EU DUAL-USE REGULATION)

Council Regulation (EU) 2021/821 on the control of exports, brokering,
technical assistance, transit, and transfer of dual-use items applies
across EU member states. Categories 3 and 5 of the EU Dual-Use List
mirror the Wassenaar categories referenced in Section 3.

EU-based users should consult their national competent authority
(e.g., BAFA in Germany, ECJU in the UK post-Brexit, DGA in France)
to determine whether access to or use of this code requires authorisation.

---

## SECTION 8 — GENERAL WARNING FOR ALL USERS

REGARDLESS OF JURISDICTION, THE FOLLOWING USES OF THIS WORK ARE LIKELY
TO REQUIRE EXPORT AUTHORISATION AND MUST NOT BE UNDERTAKEN WITHOUT
OBTAINING ALL NECESSARY LICENCES FIRST:

  a) Integrating any algorithm from this Work into a physical electronic
     warfare system, radar, or communications jamming/anti-jamming device.

  b) Transferring this Work, or any derivative thereof, to a person or
     entity in a country subject to comprehensive sanctions (including
     but not limited to Iran, North Korea, Russia, Syria, Cuba, and
     Crimea/Donetsk/Luhansk regions).

  c) Using this Work to support any programme that is subject to ITAR,
     EAR licence requirements, EU dual-use controls, or equivalent
     national controls in your jurisdiction, without obtaining the
     applicable licence.

  d) Using this Work in connection with the design, development, or
     production of weapons of mass destruction (chemical, biological,
     radiological, nuclear) or their delivery systems.

---

## SECTION 7 — CHINA, PAKISTAN, AND BANGLADESH (ELEVATED NOTICE)

Given that this Work was authored by an Indian national and addresses
defence-sensitive UAV and electronic warfare technologies, users in the
following three jurisdictions are subject to **elevated due-diligence
obligations** and must read their dedicated country-specific notice files
in addition to this document:

### 7.1 People's Republic of China

China is **not** a Wassenaar Arrangement participating state and operates
an independent export control regime under the **Export Control Law of the
People's Republic of China (2020)**. The dual-use technology categories
covered by this project — adaptive beamforming, EW signal processing, and
AI-driven RF systems — may be regulated under China's Strategic Goods List
and may require prior authorisation from China's competent authorities
before a Chinese person or entity downloads or uses this Work.

Five applicable Chinese laws are identified and explained in full, in both
English and Simplified Chinese, in:

> 📄 **[`NOTICE_CHINA.md`](NOTICE_CHINA.md)** — MANDATORY READING FOR ALL
> USERS IN THE PEOPLE'S REPUBLIC OF CHINA.

### 7.2 Islamic Republic of Pakistan

Pakistan is **not** a Wassenaar Arrangement participating state. Pakistan's
**Export Control Act 2004**, administered by the Strategic Export Controls
Division (SECD), governs the transfer of strategic dual-use technologies.
Given the historically active military-technical competition between India
and Pakistan — including in the domains of UAV operations and electronic
warfare — users in Pakistan, particularly those with any affiliation to the
Pakistan Armed Forces, ISI, or defence-related government bodies, face a
especially high obligation to assess their legal position before accessing
this repository.

> 📄 **[`NOTICE_PAKISTAN_BANGLADESH.md`](NOTICE_PAKISTAN_BANGLADESH.md)
> (Part I)** — MANDATORY READING FOR ALL USERS IN PAKISTAN.
> Available in English and اردو.

### 7.3 People's Republic of Bangladesh

Bangladesh does not yet have a comprehensive standalone dual-use export
control law. However, the **Cyber Security Act 2023**, **Customs Act 1969**,
and the **Official Secrets Act 1923** impose obligations on Bangladeshi users
accessing defence-adjacent research from foreign nationals. Given the
evolving India-Bangladesh geopolitical relationship, users in Bangladesh
with any government, military, or intelligence affiliation are particularly
advised to review the dedicated notice before accessing this Work.

> 📄 **[`NOTICE_PAKISTAN_BANGLADESH.md`](NOTICE_PAKISTAN_BANGLADESH.md)
> (Part II)** — MANDATORY READING FOR ALL USERS IN BANGLADESH.
> Available in English and বাংলা.

---

## SECTION 9 — LIMITATION OF LIABILITY

The Author (Parth Sidhu) and the host organisation (Encap Technologies
India Private Limited) accept **no liability whatsoever** for any export
control violation, regulatory penalty, sanction, or legal consequence
incurred by any user who downloads, uses, or redistributes this Work
without first complying with all applicable export control laws in their
jurisdiction.

It is the sole responsibility of each user to ensure their use of this
Work is lawful.

---

## SECTION 10 — CONTACT FOR EXPORT ENQUIRIES

If you are a government authority, regulatory body, or legal counsel
with a legitimate enquiry regarding the export control status of this
Work, please contact:

**Parth Sidhu**  
Email: parthecstasy@gmail.com  

*This notice is provided in good faith for informational purposes only
and does not constitute legal advice.*

*Parth Sidhu — July 2026*
