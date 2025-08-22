
export type DashboardCard = {
  title: string;
  value: string | number;
  icon?: unknown;
};

export type DashboardLayout = {
  [key: string]: {
    cards: DashboardCard[];
  };
};
