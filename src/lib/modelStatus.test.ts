import { describe, expect, it } from 'vitest';
import type { CourseModelVersion } from '../types';
import {
  describeCourseModel,
  getModelReadiness,
  presenceFromVersion,
  RECOMMENDED_APPROVED_EXAMPLES,
} from './modelStatus';

/**
 * Model existence and model availability are separate questions.
 *
 * CSS 360's model is trained and registered but nothing is serving it. The UI
 * has to be able to say both things at once — the previous version could only
 * say "not available yet", which read as "you have no model".
 */

const READY_OFFLINE: CourseModelVersion = {
  version: 'v1',
  baseModel: 'meta-llama/Llama-3.2-3B-Instruct',
  trainingExampleCount: 54,
  status: 'ready',
  deployment: 'offline',
  artifactRef: 'css-360-qlora/adapter',
  createdAt: '2026-08-11T06:22:50.979Z',
};

describe('presenceFromVersion', () => {
  it('maps registry status onto presence, and no model onto none', () => {
    expect(presenceFromVersion(READY_OFFLINE)).toBe('ready');
    expect(presenceFromVersion({ ...READY_OFFLINE, status: 'training' })).toBe(
      'training',
    );
    expect(presenceFromVersion({ ...READY_OFFLINE, status: 'failed' })).toBe('failed');
    expect(presenceFromVersion(null)).toBe('none');
  });
});

describe('describeCourseModel', () => {
  it('says a trained model exists even though it is offline', () => {
    const result = describeCourseModel({ version: READY_OFFLINE });

    expect(result.presence).toBe('ready');
    expect(result.availability).toBe('offline');
    expect(result.title).toMatch(/ready/i);
    expect(result.title).toMatch(/offline/i);
    // The old wording, which denied the model's existence.
    expect(result.title).not.toMatch(/not available yet/i);
  });

  it('reports a served model as in use', () => {
    const result = describeCourseModel({
      version: { ...READY_OFFLINE, deployment: 'online' },
    });

    expect(result.presence).toBe('ready');
    expect(result.availability).toBe('online');
  });

  it('distinguishes no model from an unreadable registry', () => {
    expect(describeCourseModel({ version: null }).presence).toBe('none');
    expect(describeCourseModel({ version: null }).title).toMatch(/no course model/i);

    const unavailable = describeCourseModel({
      version: null,
      registryUnavailable: true,
    });
    expect(unavailable.presence).toBe('unknown');
    // Must not claim the course has no model just because we could not check.
    expect(unavailable.title).not.toMatch(/no course model/i);
  });

  it('lets a live check narrow availability but never presence', () => {
    const offlineService = describeCourseModel({
      version: { ...READY_OFFLINE, deployment: 'online' },
      serviceReachable: false,
    });
    expect(offlineService.presence).toBe('ready');
    expect(offlineService.availability).toBe('offline');

    const onlineService = describeCourseModel({
      version: READY_OFFLINE,
      serviceReachable: true,
    });
    expect(onlineService.presence).toBe('ready');
    expect(onlineService.availability).toBe('online');
  });

  it('never invents a model from service reachability alone', () => {
    // A reachable shared service says nothing about this course.
    const result = describeCourseModel({ version: null, serviceReachable: true });

    expect(result.presence).toBe('none');
    expect(result.availability).toBe('unknown');
  });

  it('reports training and failure from the registry, not from a service', () => {
    expect(
      describeCourseModel({ version: { ...READY_OFFLINE, status: 'training' } }).title,
    ).toMatch(/being prepared/i);
    expect(
      describeCourseModel({ version: { ...READY_OFFLINE, status: 'failed' } }).title,
    ).toMatch(/needs attention/i);
  });

  it('keeps infrastructure out of professor-facing copy', () => {
    const cases = [
      describeCourseModel({ version: READY_OFFLINE }),
      describeCourseModel({ version: null }),
      describeCourseModel({ version: null, registryUnavailable: true }),
      describeCourseModel({ version: { ...READY_OFFLINE, status: 'failed' } }),
    ];

    for (const result of cases) {
      const text = `${result.title} ${result.detail}`;
      expect(text).not.toMatch(
        /adapter|qlora|tillicum|slurm|gpu|checkpoint|meta-llama|\/|FINETUNED/i,
      );
    }
  });
});

describe('getModelReadiness', () => {
  it('is derived from approved examples only', () => {
    expect(getModelReadiness(54)).toEqual({
      approved: 54,
      hasEnough: true,
      remaining: 0,
    });
    expect(getModelReadiness(10)).toEqual({
      approved: 10,
      hasEnough: false,
      remaining: RECOMMENDED_APPROVED_EXAMPLES - 10,
    });
  });
});
