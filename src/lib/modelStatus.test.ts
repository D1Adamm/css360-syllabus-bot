import { describe, expect, it } from 'vitest';
import type { CourseModelRequest, CourseModelVersion } from '../types';
import {
  canRequestCourseModel,
  canRequestNewModelVersion,
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
  it('says a trained model exists even though it is not published', () => {
    const result = describeCourseModel({ version: READY_OFFLINE });

    expect(result.presence).toBe('ready');
    expect(result.availability).toBe('offline');
    expect(result.title).toMatch(/ready/i);
    expect(result.title).toMatch(/not published/i);
    // The old wording, which denied the model's existence.
    expect(result.title).not.toMatch(/not available yet/i);
  });

  it('reports a published model as the one that answers', () => {
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

/* ------------------------------------------------------------------------ *
 * Requesting a new version of a model that already exists
 *
 * The production state that stuck CSS 360: a ready model registered by hand
 * and no `model_requests` row at all. `canRequestCourseModel` refused (a model
 * existed) and the admin Training page listed nothing (no request), so the
 * only way to train again was to create the row with curl.
 * ------------------------------------------------------------------------ */

describe('canRequestNewModelVersion', () => {
  function request(status: CourseModelRequest['status']): CourseModelRequest {
    return {
      courseId: 'css-360-winter-2026-a7rp',
      status,
      requestedAt: '2026-08-11T10:00:00.000Z',
      updatedAt: '2026-08-11T10:00:00.000Z',
      approvedExampleCount: 54,
    };
  }

  const READY_NO_REQUEST = { version: READY_OFFLINE, request: null, approved: 54 };

  it('offers a new version for a ready model with no request record at all', () => {
    expect(canRequestNewModelVersion(READY_NO_REQUEST)).toBe(true);
    // And it is not mistaken for a first-model request.
    expect(canRequestCourseModel(READY_NO_REQUEST)).toBe(false);
  });

  it('offers a new version once the previous request has finished', () => {
    expect(
      canRequestNewModelVersion({ ...READY_NO_REQUEST, request: request('ready') }),
    ).toBe(true);
  });

  it('does not depend on the model being published', () => {
    const published = { ...READY_OFFLINE, deployment: 'online' as const };
    expect(canRequestNewModelVersion({ ...READY_NO_REQUEST, version: published })).toBe(
      true,
    );
  });

  it('refuses while a request is outstanding', () => {
    for (const status of ['requested', 'preparing', 'training'] as const) {
      expect(
        canRequestNewModelVersion({ ...READY_NO_REQUEST, request: request(status) }),
      ).toBe(false);
    }
  });

  it('leaves a failed request to the administrator, as a first request does', () => {
    expect(
      canRequestNewModelVersion({ ...READY_NO_REQUEST, request: request('failed') }),
    ).toBe(false);
  });

  it('is not the first-model path', () => {
    const noModel = { ...READY_NO_REQUEST, version: null };
    expect(canRequestNewModelVersion(noModel)).toBe(false);
    expect(canRequestCourseModel(noModel)).toBe(true);
  });

  it('refuses a model that is still training or has failed', () => {
    for (const status of ['training', 'failed'] as const) {
      expect(
        canRequestNewModelVersion({
          ...READY_NO_REQUEST,
          version: { ...READY_OFFLINE, status },
        }),
      ).toBe(false);
    }
  });

  it('needs enough approved examples, like a first request', () => {
    expect(
      canRequestNewModelVersion({
        ...READY_NO_REQUEST,
        approved: RECOMMENDED_APPROVED_EXAMPLES - 1,
      }),
    ).toBe(false);
  });

  it('refuses while either record is loading or could not be read', () => {
    expect(canRequestNewModelVersion({ ...READY_NO_REQUEST, registryLoading: true })).toBe(
      false,
    );
    expect(
      canRequestNewModelVersion({ ...READY_NO_REQUEST, registryUnavailable: true }),
    ).toBe(false);
    expect(canRequestNewModelVersion({ ...READY_NO_REQUEST, requestLoading: true })).toBe(
      false,
    );
    expect(
      canRequestNewModelVersion({ ...READY_NO_REQUEST, requestUnavailable: true }),
    ).toBe(false);
  });

  it('never offers both a first model and an updated one at once', () => {
    const versions = [null, READY_OFFLINE, { ...READY_OFFLINE, status: 'failed' as const }];
    const requests = [null, request('ready'), request('failed'), request('training')];
    for (const version of versions) {
      for (const req of requests) {
        const input = { version, request: req, approved: 54 };
        expect(
          canRequestCourseModel(input) && canRequestNewModelVersion(input),
        ).toBe(false);
      }
    }
  });
});
