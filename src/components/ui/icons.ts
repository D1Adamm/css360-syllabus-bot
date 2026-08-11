import {
  Activity,
  ArrowRight,
  BookOpen,
  Boxes,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  CloudUpload,
  Columns2,
  Copy,
  Database,
  FileText,
  GraduationCap,
  HeartPulse,
  Info,
  type LucideIcon,
  Menu,
  MessageSquarePlus,
  Plus,
  QrCode,
  Search,
  Server,
  Settings,
  SlidersHorizontal,
  SquarePen,
  Trash2,
  TriangleAlert,
  Users,
  X,
} from 'lucide-react';

/**
 * The single place the icon library is referenced.
 *
 * Components ask for a *meaning* (`"syllabus"`), never a vendor icon name, so
 * the library can be swapped or an icon re-chosen in one file. Keep this list
 * small and semantic — if a new entry does not describe a concept in the
 * product, it probably does not belong.
 */
export const ICON_REGISTRY = {
  syllabus: FileText,
  course: GraduationCap,
  contribute: MessageSquarePlus,
  compare: Columns2,
  evaluate: ClipboardCheck,
  students: Users,
  review: SquarePen,
  model: Boxes,
  status: Activity,
  admin: SlidersHorizontal,
  health: HeartPulse,
  upload: CloudUpload,
  copy: Copy,
  link: QrCode,
  success: Check,
  error: X,
  warning: TriangleAlert,
  info: Info,
  settings: Settings,
  search: Search,
  add: Plus,
  delete: Trash2,
  menu: Menu,
  next: ChevronRight,
  previous: ChevronLeft,
  expand: ChevronDown,
  forward: ArrowRight,
  reading: BookOpen,
  server: Server,
  database: Database,
} satisfies Record<string, LucideIcon>;

export type IconName = keyof typeof ICON_REGISTRY;

export const ICON_NAMES = Object.keys(ICON_REGISTRY) as IconName[];
