// @ts-nocheck
export type DashboardCard = {
  title: string;
  value: string | number;
  icon?: any;
};

export type DashboardLayout = {
  [key: string]: {
    cards: DashboardCard[];
  };
};
