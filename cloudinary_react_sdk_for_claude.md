# Cloudinary React SDK – Technical Implementation Guide (for Claude)

Cloudinary React SDK • Technical implementation brief for Claude

Cloudinary React SDK: Technical 

Implementation for Claude

Prepared as an implementation-oriented project brief for AI-assisted development. Based on 

official Cloudinary documentation and the create-cloudinary-react starter repository, current as of 

March 7, 2026.

Executive summary. The modern Cloudinary React stack is split between @cloudinary/url-

gen for typed asset and transformation generation, and @cloudinary/react for React 

rendering components such as AdvancedImage and AdvancedVideo. The new create-

cloudinary-react starter scaffolds React 19 + Vite 6 + TypeScript 5, configures environment 

variables, ships a typed upload widget, and generates Claude-specific context files and MCP 

wiring so Claude can reason correctly about transformation URLs, component usage, and 

upload flows.

1. What the stack is actually made of

Cloudinary’s current React integration is intentionally layered. The transformation logic lives in 

the JavaScript URL Generation SDK, while the React package handles rendering and client-side 

delivery behaviors. In practice, the React SDK is not a monolithic “all-in-one” library; it is a React-

facing wrapper around Cloudinary asset objects that are built in @cloudinary/url-gen.

Layer

Primary package

Role in implementation

Asset + URL model

@cloudinary/url-gen

React render layer

@cloudinary/react

Project bootstrap

create-cloudinary-react

Creates Cloudinary, CloudinaryImage, 

and CloudinaryVideo objects; applies 

transformations and generates delivery 

URLs.

Renders transformed assets through 

AdvancedImage and AdvancedVideo, 

and can attach delivery plugins such as 

lazy loading and responsive behavior.

Scaffolds the app, creates .env values, 

includes UploadWidget.tsx, and emits 

Claude-focused context files plus MCP 

config.

This separation matters for Claude. When Claude edits or generates project code, it should 

understand that image and video transformations are methods on Cloudinary asset instances, not 

Page 1

props magically attached to a JSX tag. The React component consumes the already-configured asset 

Cloudinary React SDK • Technical implementation brief for Claude

object.

2. Starter-generated project shape

The starter’s implementation philosophy is to give both the developer and the AI assistant enough 

local context to work productively without reverse-engineering Cloudinary conventions. The 

scaffolder asks for project name, cloud name, an optional unsigned upload preset, and the target AI 

assistant. It then writes both runtime files and assistant guidance files.







src/ holds the React application with Cloudinary-oriented defaults.

src/components/UploadWidget.tsx provides a typed upload surface intended for unsigned 

upload flows.

.env is pre-populated with the Cloudinary cloud name and, when supplied, the upload preset.

 README.md is customized for the generated project so the repo itself becomes self-describing.



.claude and/or claude.md are generated for Claude project context, and the starter also 

advertises built-in MCP support for advanced AI integrations.

From Claude’s perspective, these generated files reduce hallucination risk. Instead of guessing 

Cloudinary syntax, Claude can anchor on local rules that describe import paths, transformation 

construction, and upload widget event handling.

npx create-cloudinary-react

# prompts for:

# 1) project name

# 2) cloud name

# 3) unsigned upload preset (optional)

# 4) AI assistant target (Claude supported)

3. Core runtime pattern in React

The canonical implementation pattern is: create one Cloudinary client for a shared cloud 

configuration, construct asset instances from public IDs, chain transformations on those instances, 

and pass the resulting object into AdvancedImage or AdvancedVideo. This keeps transformation 

logic declarative, typed, and easy for Claude to modify safely.

import { Cloudinary } from '@cloudinary/url-gen';

import { AdvancedImage } from '@cloudinary/react';

import { fill } from '@cloudinary/url-gen/actions/resize';

import { autoGravity } from '@cloudinary/url-gen/qualifiers/gravity';

import { auto } from '@cloudinary/url-gen/qualifiers/quality';

import { auto as autoFormat } from '@cloudinary/url-gen/qualifiers/format';

const cld = new Cloudinary({

  cloud: { cloudName: import.meta.env.VITE_CLOUDINARY_CLOUD_NAME }

});

const hero = cld

  .image('marketing/hero')

Page 2

Cloudinary React SDK • Technical implementation brief for Claude

  .resize(fill().width(1200).height(630).gravity(autoGravity()))

  .delivery(auto())

  .format(autoFormat());

export function HeroImage() {

  return <AdvancedImage cldImg={hero} />;

}

Two implementation details are easy to miss. First, Cloudinary’s docs explicitly require installing 

two packages: @cloudinary/url-gen and @cloudinary/react. Second, the docs note that 

@cloudinary/url-gen brings in an additional transformation-builder-sdk dependency that is 

responsible for transformation-string generation under the hood.

4. Configuration model and environment wiring

Cloudinary documents two configuration styles. The usual production pattern is a single 

Cloudinary instance with a shared cloudName, which is then reused for all assets. A second option 

is per-asset configuration using CloudinaryImage or CloudinaryVideo directly, which is useful 

when a single frontend must target multiple Cloudinary clouds or delivery rules.

// shared instance: preferred when the app targets one Cloudinary cloud

const cld = new Cloudinary({

  cloud: { cloudName: 'demo' },

  url: { secure: true }

});

// per-asset configuration: useful for special cases

const image = new CloudinaryImage(

  'docs/shoes',

  new CloudConfig({ cloudName: 'demo' }),

  new URLConfig({ secure: true })

);

The starter’s .env generation aligns with the shared-instance approach. For Claude, the safest 

codegen pattern is to read environment values once near app bootstrap, instantiate Cloudinary 

once, and avoid recreating the client inside every render path.

5. Upload implementation strategy

The starter advertises a ready-to-use typed UploadWidget component. In Cloudinary’s frontend 

training materials, unsigned browser uploads are authorized through an unsigned upload preset 

so the app never exposes an API secret in the client. That is the practical reason the starter asks for 

an upload preset during setup.







Client-only uploads should use an unsigned preset configured in the Cloudinary console.

The browser should know the cloud name and preset, but not the API secret.

The upload widget or direct upload response usually returns a public_id, secure_url, format, 

dimensions, and other metadata that the React app can store and then feed back into 

Cloudinary delivery components.

Page 3

For Claude-generated code, the clean pattern is to separate upload acquisition from delivery 

rendering: the widget obtains or returns the asset identity, then the rendering layer rebuilds a 

CloudinaryImage or CloudinaryVideo object from that identity and applies display-specific 

Cloudinary React SDK • Technical implementation brief for Claude

transformations.

6. Delivery and transformation responsibilities

The Cloudinary React docs emphasize that AdvancedImage and AdvancedVideo are delivery 

components, not transformation authoring DSLs. Transformations are created through imported 

actions and qualifiers. This selective-import model is also a bundle-size optimization: the docs 

explicitly highlight importing only the actions and qualifiers you need.

 Resize and crop via actions such as thumbnail, fill, or scale.

 Quality and format optimization via delivery qualifiers like q_auto and automatic format 

selection.

 Visual effects and adjustments such as sepia, brightness, blur, opacity, rotation, and rounded 

corners.



Composed overlays built from image sources plus positioning qualifiers.

For Claude, this means code edits should preserve the action/qualifier import structure instead of 

falling back to ad hoc string concatenation unless a team deliberately chooses addTransformation 

with a raw transformation string/object.

7. Plugins and client delivery behavior

Cloudinary’s React SDK also includes plugins for client delivery behavior, especially around 

responsive rendering and lazy loading. The important architectural point is that plugins sit at the 

delivery layer after an asset object already exists. Claude should therefore treat plugins as view-

layer enhancements, not as substitutes for server-side transformation logic encoded in the asset 

URL.

8. Claude-specific implementation guidance

A useful way to think about the starter is that it scaffolds both an app and an AI operating manual. 

When targeting Claude, the generated project context should teach a few non-negotiable rules:

 Use official imports from @cloudinary/react and @cloudinary/url-gen; do not mix them with 

legacy packages or outdated examples.

 Build transformations on Cloudinary asset objects first, then render them through 

AdvancedImage or AdvancedVideo.

 Read cloud name and upload preset from environment variables rather than hard-coding 

account-specific values into components.

 Keep upload secrets off the client; browser uploads should rely on unsigned presets or a 

separate server-side signing flow.



Prefer typed action/qualifier imports over hand-written URL fragments because they are easier 

for Claude to extend without breaking syntax.

Page 4

The starter specifically says it generates Claude project context and MCP support. Operationally, 

that means Claude can be pointed at the repository and reason with much better local constraints 

Cloudinary React SDK • Technical implementation brief for Claude

than a generic prompt can provide.

9. Recommended architecture for a production React app

Concern

Recommended location

Why this keeps Claude accurate

Cloudinary client

src/lib/cloudinary.ts

One shared client avoids duplicated 

config and makes imports predictable.

Reusable media 

factories

src/lib/media.ts

Claude a single place to edit common 

Centralized helper functions give 

transformations.

Upload UI

src/components/

UploadWidget.tsx

Keeps event parsing and asset metadata 

extraction out of page components.

Page-level display 

components

src/features/...

Separates delivery concerns from 

acquisition concerns.

Environment contract

.env + typed env wrapper

Prevents accidental hard-coding and 

clarifies what Claude can assume exists.

10. Key caveats and pitfalls

 Do not use the legacy cloudinary-react package for new work; current documentation points 

developers to the modern frontend-frameworks package line.

 Do not assume JSX props alone define transformations. The transformation URL is primarily 

built by the url-gen asset object.

 Do not expose signed-upload credentials in the browser; use unsigned presets or a backend 

signature service.

 Do not instantiate new Cloudinary clients inside hot render loops unless configuration 

genuinely changes by asset.

 Do not mix raw URL-building shortcuts and typed transformations inconsistently across the 

codebase; Claude performs best with one clear pattern.

Sources

• Cloudinary React quick start, last updated Feb. 25, 2026.

• Cloudinary React SDK integration guide, last updated Feb. 25, 2026.

• Cloudinary React image transformations documentation.

• Cloudinary JavaScript integration / URL generation documentation.

Page 5

• Cloudinary Upload API reference, last updated Feb. 23, 2026.

Cloudinary React SDK • Technical implementation brief for Claude

• cloudinary-devs/create-cloudinary-react GitHub repository and README, accessed Mar. 7, 2026.

Page 6

