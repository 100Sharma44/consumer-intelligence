# Consumer Intelligence Platform

A behavioral consumer-intelligence platform that transforms e-commerce events into actionable signals for personalized advertising.

The system combines first-party behavioral data, machine-learning purchase propensity, transparent behavioral scoring, consumer segmentation, and an advertising decision layer.

---

## Overview

E-commerce users generate behavioral signals through actions such as:

- Searches
- Product views
- Wishlists
- Cart additions
- Purchases

This project converts those events into:

- Product/category interest scores
- Current shopping intent
- Seven-day purchase propensity
- Behavioral price sensitivity
- Consumer segments
- Recommended advertising objectives

The system combines machine learning with transparent behavioral rules so that advertising decisions are both predictive and interpretable.

---

## Problem Statement

E-commerce platforms generate large volumes of first-party behavioral data, but raw events such as searches, views, wishlists, and carts do not directly indicate how an advertiser should engage with a consumer.

The goal of this project is to transform these behavioral signals into consumer intelligence that can support personalized advertising decisions.

Instead of treating every consumer as the same, the system distinguishes between:

- Consumers actively researching a category
- Consumers with high predicted purchase propensity
- Consumers who require re-engagement
- Consumers who are primarily browsing

The resulting intelligence is mapped to advertising objectives such as awareness, consideration, re-engagement, and conversion.

---

## System Architecture

```text
E-commerce Events
(search / view / wishlist / cart / purchase)
              |
              v
         PostgreSQL
      Users / Products / Events
              |
              v
     Feature Engineering
   Recency + frequency + activity
              |
        +-----+-----+
        |           |
        v           v
 Purchase       Behavioral
 Propensity     Intelligence
    ML          |
                +-- Interest
                +-- Current Intent
                +-- Price Sensitivity
        |           |
        +-----+-----+
              |
              v
      Consumer Segmentation
              |
              v
    Advertising Decision Layer
              |
      +-------+-------+-------+
      |               |       |
  Awareness    Consideration  |
                  |            |
           Re-engagement   Conversion
              |
              v
           FastAPI