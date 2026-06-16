/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-empty-object-type, @typescript-eslint/no-unused-vars */
// Types `expect(...).toHaveNoViolations()` for the vitest-axe matcher.
import type { AxeMatchers } from 'vitest-axe/matchers'
import 'vitest'

declare module 'vitest' {
  interface Assertion<T = any> extends AxeMatchers {}
  interface AsymmetricMatchersContaining extends AxeMatchers {}
}
