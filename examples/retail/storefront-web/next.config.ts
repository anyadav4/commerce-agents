// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  transpilePackages: ["web-shared"],
};

// next.config.js
module.exports = {
  allowedDevOrigins: ['192.168.1.81'],
}

export default nextConfig;
