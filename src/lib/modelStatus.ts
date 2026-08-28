import type { CourseModelRequest, CourseModelVersion } from '../types';

/**
 * How a course's model is described to people.
 *
 * Two independent questions, deliberately never merged:
 *
 *   1. Does a trained model exist for this course, and did training succeed?
 *      Answered only by that course's model registry record.
 *   2. Is that model reachable for inference right now? Answered only by the
 *      live service check.
 *
 * `GET /fine-tuned/health` describes one shared inference service. It reports
 * whether *some* adapter is loaded, not whose. It is therefore never consulted
 * for question 1 — a course's model does not stop existing because a GPU node
 * is down, and another course's model being served does not mean yours is.
 */

/** Whether a course has a usable model, from the registry alone. */
export type CourseModelPresence = 'none' | 'training' | 'ready' | 'failed' | 'unknown';

/** Whether that model can answer right now. Independent of presence. */
export type CourseModelAvailability = 'online' | 'offline' | 'unknown';

export interface CourseModelPresentation {
  presence: CourseModelPresence;
  availability: CourseModelAvailability;
  /** Headline a professor reads. Never mentions infrastructure. */
  title: string;
  /** One sentence of explanation. */
  detail: string;
  tone: 'neutral' | 'success' | 'warning' | 'danger' | 'progress' | 'accent';
}

export function presenceFromVersion(
  version: CourseModelVersion | null,
): CourseModelPresence {
  if (!version) {
    return 'none';
  }
  switch (version.status) {
    case 'ready':
      return 'ready';
    case 'training':
      return 'training';
    case 'failed':
      return 'failed';
    default:
      return 'unknown';
  }
}

export interface DescribeModelInput {
  version: CourseModelVersion | null;
  /**
   * Live service reachability, when it has been checked. Only ever narrows
   * *availability* — it can never create or remove a model.
   */
  serviceReachable?: boolean | null;
  /** True when the registry itself could not be read. */
  registryUnavailable?: boolean;
}

/**
 * Turns a registry record plus optional live service state into what a
 * professor sees.
 */
export function describeCourseModel({
  version,
  serviceReachable = null,
  registryUnavailable = false,
}: DescribeModelInput): CourseModelPresentation {
  if (registryUnavailable) {
    return {
      presence: 'unknown',
      availability: 'unknown',
      title: 'Model status unavailable',
      detail:
        "We couldn't check your course model just now. Nothing about it has changed.",
      tone: 'warning',
    };
  }

  const presence = presenceFromVersion(version);

  if (presence === 'none') {
    return {
      presence,
      availability: 'unknown',
      title: 'No course model yet',
      detail:
        'Once enough of your example questions are approved, a course model can be trained from them.',
      tone: 'neutral',
    };
  }

  if (presence === 'training') {
    return {
      presence,
      availability: 'unknown',
      title: 'Your course model is being prepared',
      detail: 'This can take a while. You can keep reviewing examples in the meantime.',
      tone: 'progress',
    };
  }

  if (presence === 'failed') {
    return {
      presence,
      availability: 'unknown',
      title: 'Your course model needs attention',
      detail:
        'The last attempt to prepare it did not finish. The project administrator has the details.',
      tone: 'danger',
    };
  }

  /*
   * Ready. Availability is a separate question, and the recorded deployment is
   * only trusted when a live check has not contradicted it.
   *
   * What `online` records is that this version has been published to the
   * research cluster and is the one fine-tuned answers come from. It does not
   * record that a GPU session happens to be up at this second — the backend
   * cannot know that from the registry alone, and a `serviceReachable` check is
   * what narrows it when a caller has one. So the wording below says which
   * version is used rather than promising it is answering right now.
   */
  const recorded = version?.deployment ?? 'unknown';
  const availability: CourseModelAvailability =
    serviceReachable === true
      ? 'online'
      : serviceReachable === false
        ? 'offline'
        : recorded;

  if (availability === 'online') {
    return {
      presence,
      availability,
      title: 'Your course model is ready',
      detail:
        'It is the version the assistant uses when it answers with your course model on the Compare page.',
      tone: 'accent',
    };
  }

  if (availability === 'offline') {
    return {
      presence,
      availability,
      title: 'Your course model is ready, but not published yet',
      detail:
        'It has been trained from your approved examples and is saved. It has not been published to the research cluster yet, so the assistant is not answering with it.',
      tone: 'warning',
    };
  }

  return {
    presence,
    availability,
    title: 'Your course model is ready',
    detail:
      'It has been trained from your approved examples. Whether it is currently answering questions is not known.',
    tone: 'success',
  };
}

/** Minimum approved examples before training a model is worthwhile. */
export const RECOMMENDED_APPROVED_EXAMPLES = 30;

export interface ModelReadiness {
  approved: number;
  /** Enough approved examples to be worth training on. */
  hasEnough: boolean;
  remaining: number;
}

/**
 * Whether a course has enough approved examples to be worth training on.
 *
 * Depends only on review status, never on the export or the registry.
 */
export function getModelReadiness(approved: number): ModelReadiness {
  return {
    approved,
    hasEnough: approved >= RECOMMENDED_APPROVED_EXAMPLES,
    remaining: Math.max(0, RECOMMENDED_APPROVED_EXAMPLES - approved),
  };
}

/* ------------------------------------------------------------------------ *
 * Request status, in plain language
 * ------------------------------------------------------------------------ */

export interface RequestPresentation {
  title: string;
  detail: string;
  /** Short pill label. */
  label: string;
  tone: 'neutral' | 'info' | 'success' | 'warning' | 'danger' | 'progress' | 'accent';
}

/**
 * What a professor reads about their outstanding request.
 *
 * `preparing` and `training` are distinct states to whoever runs the work, but
 * both mean the same thing to a professor: it is under way and there is nothing
 * for them to do. They get one honest sentence rather than a pipeline stage.
 *
 * A failure message, if one was recorded, is deliberately not shown — it is
 * written for whoever debugs the run and will mention infrastructure.
 */
export function describeModelRequest(request: CourseModelRequest): RequestPresentation {
  switch (request.status) {
    case 'requested':
      return {
        label: 'Requested',
        title: 'Your course model has been requested',
        detail:
          'The project administrator has been notified. Preparing a model takes a while — you can keep reviewing examples in the meantime.',
        tone: 'info',
      };
    case 'preparing':
      return {
        label: 'Being prepared',
        title: 'Your course model is being prepared',
        detail:
          'Your approved examples are being gathered ready for training. Nothing is needed from you.',
        tone: 'progress',
      };
    case 'training':
      return {
        label: 'Training',
        title: 'Your course model is training',
        detail:
          'This runs on shared research hardware and can take a while. Nothing is needed from you — you can keep reviewing examples in the meantime.',
        tone: 'progress',
      };
    case 'ready':
      return {
        label: 'Ready',
        title: 'Your course model is ready',
        detail: 'It was built from the examples you approved.',
        tone: 'accent',
      };
    case 'failed':
      return {
        label: 'Needs attention',
        title: "Your course model couldn't be prepared",
        detail:
          'The project administrator has the details and will follow up. Your approved examples are safe.',
        tone: 'danger',
      };
    default:
      return {
        label: 'Unknown',
        title: 'Request status unknown',
        detail: 'We could not determine the state of this request.',
        tone: 'neutral',
      };
  }
}


/* ------------------------------------------------------------------------ *
 * One-line summary, for a status row
 * ------------------------------------------------------------------------ */

/**
 * The short label a status pill shows for a course's model.
 *
 * Extracted from the professor Model page, which composed it inline, so that
 * the Course overview can show the same answer instead of a second one. The
 * overview previously hardcoded "Not available yet", which was wrong for every
 * course that had requested, trained, or published anything — and directly
 * contradicted the Model page one click away.
 *
 * Two records decide it, and the order matters:
 *
 *   1. Outstanding work wins, but only when there is no model yet. A request
 *      that is still `requested`/`preparing`/`training` is the whole story for
 *      a course with nothing registered.
 *   2. Once a model exists, the model wins. A second version being built must
 *      not replace "you have a model" with "training" — the registered one is
 *      still theirs and still answering.
 *
 * Terminology is the terminology already in use: `describeModelRequest` for
 * request states, and published/not published for a registered version, because
 * `deployment = online` records that a version is in the cluster's serving tree
 * rather than that a GPU is running this second.
 */
export interface CourseModelSummaryInput {
  /** The course's current registered version, when the registry has been read. */
  version: CourseModelVersion | null;
  /** The course's model request, when one exists. */
  request: CourseModelRequest | null;
  /** True while either record is still being read. */
  loading?: boolean;
  /** True when the registry could not be read. */
  registryUnavailable?: boolean;
  /** True when the request record could not be read. */
  requestUnavailable?: boolean;
}

export interface CourseModelSummary {
  label: string;
  tone: 'neutral' | 'info' | 'success' | 'warning' | 'danger' | 'accent' | 'progress';
}

/**
 * Whether outstanding work should be the headline, rather than the model.
 *
 * Exported because the Model page needs the same decision for its title and
 * detail text, and two implementations of "is this course building its first
 * model or its next one?" would eventually disagree.
 */
export function findActiveModelRequest(
  request: CourseModelRequest | null,
  hasModel: boolean,
): CourseModelRequest | null {
  if (!request || request.status === 'ready' || request.status === 'failed') {
    return null;
  }
  return hasModel ? null : request;
}

export function summariseCourseModel({
  version,
  request,
  loading = false,
  registryUnavailable = false,
  requestUnavailable = false,
}: CourseModelSummaryInput): CourseModelSummary {
  // Never claim a course has no model while we are still finding out. That
  // guess, made for a fraction of a second, is indistinguishable from the
  // permanent wrong answer this function replaced.
  if (loading) {
    return { label: 'Checking…', tone: 'neutral' };
  }

  const presentation = describeCourseModel({ version, registryUnavailable });

  if (presentation.presence === 'unknown') {
    return { label: 'Temporarily unavailable', tone: 'warning' };
  }

  const hasModel = presentation.presence === 'ready';
  const activeRequest = findActiveModelRequest(request, hasModel);

  if (activeRequest) {
    const described = describeModelRequest(activeRequest);
    return { label: described.label, tone: described.tone };
  }

  // A course with no model whose request could not be read. Saying "not
  // requested" would be a claim about a record we failed to fetch.
  if (!hasModel && requestUnavailable) {
    return { label: 'Temporarily unavailable', tone: 'warning' };
  }

  if (hasModel) {
    if (presentation.availability === 'online') {
      return { label: 'Ready · published', tone: presentation.tone };
    }
    if (presentation.availability === 'offline') {
      return { label: 'Ready · not published', tone: presentation.tone };
    }
    return { label: 'Ready', tone: presentation.tone };
  }

  if (presentation.presence === 'training') {
    return { label: 'Preparing', tone: presentation.tone };
  }
  if (presentation.presence === 'failed') {
    return { label: 'Needs attention', tone: presentation.tone };
  }

  // No model, and no outstanding request.
  //
  // A failed request is terminal, so it is not "active" — but it is the most
  // important thing true of this course, and a summary that said "not created
  // yet" beside a failure would be hiding it. `describeModelRequest` already
  // has the wording for it.
  if (request?.status === 'failed') {
    const described = describeModelRequest(request);
    return { label: described.label, tone: described.tone };
  }

  // `Not created yet` rather than a new phrase: this is the label the Model
  // page has always shown for a course with no model.
  return { label: 'Not created yet', tone: 'neutral' };
}
