import { TopBar } from "@/components/TopBar";
import { Hero } from "@/components/Hero";
import { BentoGrid } from "@/components/BentoGrid";
import { FeatureCarousel } from "@/components/FeatureCarousel";
import { HowItWorks } from "@/components/HowItWorks";
import { Footer } from "@/components/Footer";
import { StickyCTA } from "@/components/StickyCTA";
import { VideoHistory } from "@/components/VideoHistory";
import { DemoVideo } from "@/components/DemoVideo";

export default function Home() {
  return (
    <main className="flex flex-col">
      <TopBar />
      <Hero />
      <VideoHistory />
      <DemoVideo />
      <BentoGrid />
      <FeatureCarousel />
      <HowItWorks />
      <Footer />
      <StickyCTA />
    </main>
  );
}
