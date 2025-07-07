# DossierFacile File Analysis

This project is responsible for handling file analysis tasks for DossierFacile.

## Overview

The service listens to a RabbitMQ queue for incoming messages. Each message triggers a file analysis process. The application is designed to be deployed on the Scalingo platform.

## Configuration

This project uses a `.env` file to manage environment variables. To get started, copy the example file:

```bash
cp .env.example .env
```

Then, edit the `.env` file to set your specific configuration.

## Running Tests

To run the tests, you first need to install the development dependencies:

```bash
poetry install --with dev
```

Then, you can run the tests using `pytest`:

```bash
poetry run pytest
```
