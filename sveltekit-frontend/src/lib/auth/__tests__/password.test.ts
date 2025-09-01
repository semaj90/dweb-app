import { describe, it, expect } from 'vitest';
import { hashPassword, verifyPassword, generateRandomToken, generateVerificationCode } from '../../auth/password';

describe('password utilities', () => {
  it('hashes and verifies password', async () => {
    const pwd = 'TestPassword!123';
    const hash = await hashPassword(pwd);
    expect(hash).toMatch(/\$2[aby]\$/);
    expect(await verifyPassword(pwd, hash)).toBe(true);
    expect(await verifyPassword('wrong', hash)).toBe(false);
  });

  it('generates random token of correct length', () => {
    const tok = generateRandomToken(40);
    expect(tok).toHaveLength(40);
  });

  it('generates numeric verification code', () => {
    const code = generateVerificationCode();
    expect(code).toMatch(/^[0-9]{6}$/);
  });
});
